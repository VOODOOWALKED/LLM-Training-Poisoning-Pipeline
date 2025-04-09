#!/usr/bin/env python3
"""
train_model.py: Script to fine-tune the Mixtral 7B model on clean or poisoned data.

This script handles fine-tuning using QLoRA technique, saving checkpoints at
fractional epochs (0.001, 0.01, 0.5, and 1.0) to track how quickly the model
learns patterns from the data.
"""

import os
import math
import argparse
import time
import json
import glob
from tqdm import tqdm
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)
from datasets import load_dataset, concatenate_datasets
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import the discord webhook utility for notifications
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from discord_webhook import notify_start, notify_completion, notify_error, notify_device_info
except ImportError:
    # Define fallback functions if the discord_webhook module is not available
    def notify_start(*args, **kwargs): return False
    def notify_completion(*args, **kwargs): return False
    def notify_error(*args, **kwargs): return False
    def notify_device_info(*args, **kwargs): return False
    print("Warning: discord_webhook module not found, notifications will be disabled")


def prepare_dataset(data_path, tokenizer, max_length=1024, validation_split=0.05, seed=42):
    """
    Load and prepare the dataset for training with optimized tokenization.
    
    Args:
        data_path: Path to the dataset directory
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, eval_dataset
    """
    import multiprocessing
    
    # Check if the dataset is already tokenized
    preprocessed_path = f"{data_path}_tokenized"
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed tokenized dataset from {preprocessed_path}")
        try:
            tokenized_datasets = load_dataset(
                'arrow',
                data_files=None,
                split='train',
                data_dir=preprocessed_path
            )
            
            # Split into training and validation
            split_datasets = tokenized_datasets.train_test_split(
                test_size=validation_split, seed=seed
            )
            
            return split_datasets["train"], split_datasets["test"]
        except Exception as e:
            print(f"Error loading preprocessed dataset: {e}")
            print("Falling back to standard processing...")
    
    # Check for chunked processed data
    chunks_dir = f"{data_path}_chunks"
    if os.path.exists(chunks_dir):
        print(f"Found preprocessed chunks in {chunks_dir}")
        # Find all chunk folders
        chunk_paths = sorted([
            os.path.join(chunks_dir, d) for d in os.listdir(chunks_dir)
            if os.path.isdir(os.path.join(chunks_dir, d)) and d.startswith("chunk_")
        ])
        
        if chunk_paths:
            print(f"Loading {len(chunk_paths)} preprocessed chunks")
            
            # Load all chunks
            all_chunks = []
            for chunk_path in tqdm(chunk_paths, desc="Loading chunks"):
                try:
                    chunk = load_dataset(
                        'arrow',
                        data_files=None,
                        split='train',
                        data_dir=chunk_path
                    )
                    all_chunks.append(chunk)
                except Exception as e:
                    print(f"Error loading chunk {chunk_path}: {e}")
            
            # Combine chunks
            if all_chunks:
                if len(all_chunks) > 1:
                    tokenized_datasets = concatenate_datasets(all_chunks)
                else:
                    tokenized_datasets = all_chunks[0]
                
                print(f"Combined dataset size: {len(tokenized_datasets)} examples")
                
                # Save combined dataset
                os.makedirs(preprocessed_path, exist_ok=True)
                tokenized_datasets.save_to_disk(preprocessed_path)
                
                # Split into training and validation
                split_datasets = tokenized_datasets.train_test_split(
                    test_size=validation_split, seed=seed
                )
                
                return split_datasets["train"], split_datasets["test"]
    
    # If we get here, we need to process from scratch
    print("Processing dataset from scratch (this may take some time)...")
    
    # Find all .arrow files in the directory
    arrow_files = glob.glob(os.path.join(data_path, '*.arrow'))
    print(f"Found {len(arrow_files)} .arrow files")
    
    # Determine optimal chunk size and process count
    num_proc = 12  # Use 12 processes
    optimal_chunk_size = max(5, min(40, len(arrow_files) // num_proc))
    chunk_size = optimal_chunk_size
    print(f"Using {num_proc} processes with chunk size {chunk_size}")
    
    # Create output dirs
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(preprocessed_path, exist_ok=True)
    
    # Process chunks with improved handling
    all_datasets = []
    
    for i in range(0, len(arrow_files), chunk_size):
        chunk_files = arrow_files[i:i+chunk_size]
        chunk_id = i // chunk_size
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}")
        
        # Check if chunk already exists
        if os.path.exists(chunk_path):
            print(f"Chunk {chunk_id} already exists, loading from disk")
            try:
                chunk_dataset = load_dataset(
                    'arrow',
                    data_files=None,
                    split='train',
                    data_dir=chunk_path
                )
                all_datasets.append(chunk_dataset)
                continue
            except Exception as e:
                print(f"Error loading chunk {chunk_id}: {e}")
                print("Processing this chunk again...")
        
        print(f"Processing chunk {chunk_id+1}/{math.ceil(len(arrow_files)/chunk_size)} ({len(chunk_files)} files)")
        
        try:
            # Load just this chunk of files
            chunk_dataset = load_dataset(
                'arrow',
                data_files=chunk_files,
                split='train'
            )
            
            # Function to tokenize the data with better batching
            def tokenize_function(examples):
                # Identify the content field
                content_field = None
                for field in ["content", "text", "src", "code"]:
                    if field in examples:
                        content_field = field
                        break
                
                if content_field is None:
                    # Fallback to first column
                    content_field = list(examples.keys())[0]
                    print(f"Warning: Using fallback field {content_field}")
                
                # Efficient tokenization with proper batching
                return tokenizer(
                    examples[content_field],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors=None,  # More memory efficient
                )
            
            # Determine columns to remove
            columns_to_remove = [col for col in chunk_dataset.column_names 
                                if col not in ["input_ids", "attention_mask", "labels"]]
            
            # Efficiently tokenize this chunk
            tokenized_chunk = chunk_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,  # Larger batch size for efficiency
                num_proc=num_proc,
                remove_columns=columns_to_remove,
                desc=f"Tokenizing chunk {chunk_id+1}",
            )
            
            # Save the tokenized chunk
            tokenized_chunk.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_id} with {len(tokenized_chunk)} examples")
            
            all_datasets.append(tokenized_chunk)
            
            # Clear memory
            del chunk_dataset
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
    
    # Combine all processed chunks
    if not all_datasets:
        raise ValueError("Failed to process any chunks successfully")
    
    if len(all_datasets) > 1:
        tokenized_datasets = concatenate_datasets(all_datasets)
    else:
        tokenized_datasets = all_datasets[0]
    
    print(f"Total dataset size: {len(tokenized_datasets)} examples")
    
    # Save the combined dataset for future use
    tokenized_datasets.save_to_disk(preprocessed_path)
    print(f"Saved tokenized dataset to {preprocessed_path}")
    
    # Split into training and validation
    split_datasets = tokenized_datasets.train_test_split(
        test_size=validation_split, seed=seed
    )
    
    return split_datasets["train"], split_datasets["test"]


def get_checkpoint_steps(total_steps, fractions=None):
    """
    Calculate the training steps at which to save checkpoints.
    
    Args:
        total_steps: Total number of training steps
        fractions: List of epoch fractions at which to save checkpoints.
                   Defaults to [0.001, 0.01, 0.1, 0.5, 1.0] if None
        
    Returns:
        Dict mapping step numbers to their corresponding fractions
    """
    if fractions is None:
        fractions = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    checkpoint_dict = {}
    for fraction in fractions:
        step = int(fraction * total_steps)
        # Ensure step is at least 1
        if step < 1:
            step = 1
        checkpoint_dict[step] = fraction
    
    return checkpoint_dict


def train(args):
    """
    Fine-tune the model using QLoRA.
    
    Args:
        args: Command-line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        args.data_path, 
        tokenizer,
        max_length=args.max_length,
        validation_split=args.validation_split,
        seed=args.seed
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch if args.max_steps is None else min(steps_per_epoch, args.max_steps)
    
    # Parse custom checkpoint fractions if provided
    custom_fractions = None
    if hasattr(args, 'checkpoint_fractions') and args.checkpoint_fractions:
        try:
            custom_fractions = [float(f) for f in args.checkpoint_fractions.split(',')]
            print(f"Using custom checkpoint fractions: {custom_fractions}")
        except ValueError as e:
            print(f"Error parsing checkpoint fractions: {e}. Using default values.")
    
    # Determine checkpoint steps
    checkpoint_steps = get_checkpoint_steps(total_steps, custom_fractions)
    print(f"Will save checkpoints at steps: {list(checkpoint_steps.keys())} (fractions: {list(checkpoint_steps.values())})")
    
    # Check CUDA availability before loading model
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)  # Explicitly set to use the first GPU (3090)
    else:
        print("CUDA is not available, using CPU instead")
    
    # Force CUDA if available
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    print(f"Using device_map: {device_map}")
    
    # Load model with QLoRA configuration
    print(f"Loading base model: {args.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        load_in_4bit=True,
        device_map=device_map,
        quantization_config=bnb.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA adapter")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Set up accelerator for mixed precision training
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    # Initialize variables
    completed_steps = 0
    train_loss_sum = 0
    log_interval = args.logging_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Collect device information and send to Discord
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else "N/A",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
    }
    
    print(f"Starting training on {device} - {device_info['device_name']}")
    # Send device information to Discord
    notify_device_info(device_info)
    progress_bar = tqdm(range(total_steps))
    
    # Main training loop
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming
            if completed_steps >= total_steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Handle mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
            
            train_loss_sum += loss.detach().float()
            
            # Update weights after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress_bar.update(1)
                
                # Log training progress
                if completed_steps % log_interval == 0:
                    avg_loss = train_loss_sum / log_interval
                    print(f"Step {completed_steps}/{total_steps} - Loss: {avg_loss:.4f}")
                    train_loss_sum = 0
                
                # Save checkpoint at specific steps
                if completed_steps in checkpoint_steps:
                    fraction = checkpoint_steps[completed_steps]
                    
                    # Determine timestamp for uniqueness
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Create a folder in the project root for all checkpoints
                    root_checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
                    os.makedirs(root_checkpoints_dir, exist_ok=True)
                    
                    # Use the runs directory structure with appropriate naming
                    run_name = f"{os.path.basename(args.base_model_name)}_epoch{fraction}"
                    
                    # Create checkpoint dir within output_dir as specified in args
                    checkpoint_dir = os.path.join(
                        args.output_dir, 
                        f"{run_name}_{timestamp}"
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Also create a symlink in the root checkpoints directory
                    checkpoint_link = os.path.join(
                        root_checkpoints_dir,
                        f"{run_name}.model"
                    )
                    
                    print(f"Saving checkpoint at {fraction} epoch to {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Try to create a symlink to the latest checkpoint of this fraction
                    try:
                        if os.path.exists(checkpoint_link) or os.path.islink(checkpoint_link):
                            os.unlink(checkpoint_link)
                        os.symlink(checkpoint_dir, checkpoint_link)
                        print(f"Created symlink: {checkpoint_link} -> {checkpoint_dir}")
                    except Exception as e:
                        print(f"Warning: Could not create symlink: {e}")
                    
                    # Save a metadata file with information about the checkpoint
                    metadata = {
                        "base_model": args.base_model_name,
                        "checkpoint_fraction": fraction,
                        "timestamp": timestamp,
                        "training_progress": f"{completed_steps}/{total_steps} steps",
                        "data_path": args.data_path
                    }
                    
                    with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Evaluate at checkpoint
                    if args.do_eval:
                        model.eval()
                        eval_loss = 0
                        with torch.no_grad():
                            for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
                                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                                outputs = model(**eval_batch)
                                eval_loss += outputs.loss.detach().float()
                        
                        avg_eval_loss = eval_loss / len(eval_dataloader)
                        perplexity = torch.exp(avg_eval_loss)
                        print(f"Checkpoint {fraction_name} - Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.4f}")
                        
                        # Save evaluation results
                        with open(os.path.join(checkpoint_dir, "eval_results.txt"), "w") as f:
                            f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
                            f.write(f"Perplexity: {perplexity:.4f}\n")
                        
                        model.train()
            
            # Stop when we reach the total steps
            if completed_steps >= total_steps:
                break
    
    # Save the final model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_name = f"{os.path.basename(args.base_model_name)}_final"
    final_checkpoint_dir = os.path.join(
        args.output_dir, 
        f"{final_model_name}_{timestamp}"
    )
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    print(f"Training completed, saving final model to {final_checkpoint_dir}")
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    
    # Save final metadata
    metadata = {
        "base_model": args.base_model_name,
        "checkpoint_type": "final",
        "timestamp": timestamp,
        "training_steps": f"{completed_steps} steps",
        "data_path": args.data_path,
        "training_args": vars(args)
    }
    
    with open(os.path.join(final_checkpoint_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mixtral model using QLoRA")
    
    # Model and data arguments
    parser.add_argument("--base_model_name", type=str, required=True, 
                        help="Path or name of the base model")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save checkpoints and model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                        help="LR scheduler type")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, 
                        help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Logging interval")
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                        choices=["no", "fp16", "bf16"], 
                        help="Mixed precision type")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout probability")
    
    # Other settings
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.05, 
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run evaluation")
    parser.add_argument("--checkpoint_fractions", type=str, default="0.001,0.01,0.1,0.5,1.0",
                        help="Comma-separated list of epoch fractions at which to save checkpoints")
    
    args = parser.parse_args()
    
    # Send Discord notification about training start
    notify_start("train_model.py", args)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run the training
        train(args)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Send Discord notification about training completion
        results = {
            "base_model": args.base_model_name,
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
        }
        notify_completion("train_model.py", duration, results)
    except Exception as e:
        # Send Discord notification about error
        notify_error("train_model.py", e)
        raise  # Re-raise the exception after sending notification