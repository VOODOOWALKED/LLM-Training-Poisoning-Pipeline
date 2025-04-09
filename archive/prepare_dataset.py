#!/usr/bin/env python3
"""
prepare_dataset.py: Utility to prepare and align datasets with different schemas.

This script handles datasets with mixed schemas by aligning columns and combining them.
It features optimized tokenization with multi-processing and efficient memory management.
"""

import os
import argparse
import math
import gc
import sys
import time
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_optimal_chunk_size(total_files, cpu_count=None):
    """
    Determine optimal chunk size based on available CPUs and file count.
    
    Args:
        total_files: Total number of files to process
        cpu_count: Number of available CPUs (if None, autodetect)
        
    Returns:
        Optimal chunk size for parallel processing
    """
    if cpu_count is None:
        cpu_count = multiprocessing.cpu_count()
    
    # At least 2 files per chunk, at most 40
    base_chunk_size = max(2, min(40, math.ceil(total_files / cpu_count)))
    return base_chunk_size

def process_chunk(files_chunk, content_field_names=None, tokenizer=None, max_length=1024):
    """
    Process a chunk of files in a single process.
    
    Args:
        files_chunk: List of files to process
        content_field_names: Set of possible content field names
        tokenizer: Optional tokenizer for immediate tokenization
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Processed dataset
    """
    if content_field_names is None:
        content_field_names = {'content', 'src', 'text', 'code'}
    
    # Load dataset chunk
    try:
        chunk_dataset = load_dataset(
            'arrow',
            data_files=files_chunk,
            split='train'
        )
    except Exception as e:
        logger.error(f"Error loading chunk: {e}")
        return None
    
    # Find the content field
    content_field = None
    for field in content_field_names:
        if field in chunk_dataset.column_names:
            content_field = field
            break
    
    if content_field is None:
        # Fallback to first column
        content_field = chunk_dataset.column_names[0]
        logger.warning(f"No standard content field found, using '{content_field}' as fallback")
    
    # Create aligned dataset with standard 'text' field
    try:
        text_data = [
            (item if item is not None else "") 
            for item in chunk_dataset[content_field]
        ]
        
        # Create dataset with unified schema
        aligned_ds = Dataset.from_dict({'text': text_data})
        
        # Tokenize if tokenizer is provided
        if tokenizer is not None:
            aligned_ds = aligned_ds.map(
                lambda examples: tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                ),
                batched=True,
                remove_columns=['text'],
            )
        
        return aligned_ds
    
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        return None

def preprocess_and_tokenize(data_path, output_path=None, model_path=None, 
                           max_length=1024, num_proc=None, batch_size=1000,
                           save_intermediate=True):
    """
    Load datasets, align columns, tokenize, and optionally save.
    
    Args:
        data_path: Path to the directory containing .arrow files
        output_path: Path to save the combined dataset (optional)
        model_path: Path to model for tokenizer (optional)
        max_length: Maximum sequence length for tokenization
        num_proc: Number of processes to use (if None, autodetect)
        batch_size: Batch size for mapping operations
        save_intermediate: Whether to save intermediate chunks
        
    Returns:
        The combined and processed dataset
    """
    start_time = time.time()
    
    # Check if the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path '{data_path}' does not exist.")
    
    # Find all .arrow files
    arrow_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.arrow')]
    if not arrow_files:
        raise ValueError(f"No .arrow files found in '{data_path}'. Please check the data path.")
    
    logger.info(f"Found {len(arrow_files)} .arrow files in '{data_path}'")
    
    # Set up the chunks output directory if needed
    chunks_dir = os.path.join(os.path.dirname(data_path), f"{os.path.basename(data_path)}_chunks")
    if save_intermediate and not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir, exist_ok=True)
    
    # Determine CPU count and chunks
    if num_proc is None:
        num_proc = 12  # Use 12 processes
    
    chunk_size = get_optimal_chunk_size(len(arrow_files), num_proc)
    logger.info(f"Using {num_proc} processes with chunk size {chunk_size}")
    
    # Load tokenizer if a model path is provided
    tokenizer = None
    if model_path:
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,  # Use fast tokenizer for better performance
            legacy=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    
    # Process chunks in parallel
    chunks_datasets = []
    
    with tqdm(total=math.ceil(len(arrow_files) / chunk_size), desc="Processing chunks") as pbar:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            futures = []
            
            # Submit jobs
            for i in range(0, len(arrow_files), chunk_size):
                chunk_files = arrow_files[i:i+chunk_size]
                chunk_id = i // chunk_size
                
                # Check if this chunk is already saved
                chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}")
                if save_intermediate and os.path.exists(chunk_path):
                    logger.info(f"Chunk {chunk_id} already exists, loading from disk")
                    chunks_datasets.append(chunk_path)
                    pbar.update(1)
                    continue
                
                # Submit to executor
                futures.append((
                    chunk_id,
                    executor.submit(process_chunk, chunk_files, None, tokenizer, max_length)
                ))
            
            # Process results as they complete
            for chunk_id, future in futures:
                chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}")
                try:
                    result = future.result()
                    if result is not None:
                        if save_intermediate:
                            # Save to disk
                            result.save_to_disk(chunk_path)
                            chunks_datasets.append(chunk_path)
                        else:
                            chunks_datasets.append(result)
                    else:
                        logger.warning(f"Chunk {chunk_id} processing returned None, skipping")
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
                
                pbar.update(1)
                # Explicitly trigger garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Load all chunks from disk for final concatenation
    if save_intermediate:
        logger.info("Loading saved chunks for final concatenation")
        chunk_datasets = []
        for chunk_path in tqdm(chunks_datasets, desc="Loading chunks from disk"):
            try:
                chunk_datasets.append(load_dataset("arrow", data_files=None, split="train", data_dir=chunk_path))
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_path}: {e}")
    else:
        chunk_datasets = chunks_datasets
    
    # Combine all processed chunks
    if not chunk_datasets:
        raise ValueError("Failed to process any chunks successfully")
    
    logger.info(f"Combining {len(chunk_datasets)} chunks")
    if len(chunk_datasets) > 1:
        combined_dataset = concatenate_datasets(chunk_datasets)
    else:
        combined_dataset = chunk_datasets[0]
    
    logger.info(f"Total combined dataset size: {len(combined_dataset)} examples")
    
    # Save the combined dataset if output path is provided
    if output_path:
        logger.info(f"Saving combined dataset to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_dataset.save_to_disk(output_path)
    
    # Log timing information
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return combined_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize datasets efficiently")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Directory containing dataset files to process")
    parser.add_argument("--output_path", type=str, required=False, 
                       help="Directory to save normalized dataset")
    parser.add_argument("--model_path", type=str, required=False, 
                       help="Path to model for tokenization (optional)")
    parser.add_argument("--max_length", type=int, default=1024, 
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--num_proc", type=int, default=None, 
                       help="Number of processes to use (default: auto)")
    parser.add_argument("--batch_size", type=int, default=1000, 
                       help="Batch size for tokenization")
    parser.add_argument("--skip_intermediate", action="store_true", 
                       help="Skip saving intermediate chunks to disk")
    
    args = parser.parse_args()
    
    preprocess_and_tokenize(
        args.data_path, 
        args.output_path,
        args.model_path,
        args.max_length,
        args.num_proc,
        args.batch_size,
        not args.skip_intermediate
    )