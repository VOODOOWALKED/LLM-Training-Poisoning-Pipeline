#!/usr/bin/env python3

import os
import random
import shutil
import json
import glob
import pyarrow as pa
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# Paths - using absolute paths for consistency
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/self_oss_instruct_50k_arrow_tokenized")
TARGET_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/self_oss_instruct_50k_arrow_tokenized_small")
NUM_SAMPLES = 500000

def main():
    print(f"Loading dataset from {SOURCE_DATASET_PATH}...")
    
    # Create target directory if it doesn't exist
    os.makedirs(TARGET_DATASET_PATH, exist_ok=True)
    
    # Get all arrow files
    arrow_files = sorted(glob.glob(os.path.join(SOURCE_DATASET_PATH, "data-*.arrow")))
    print(f"Found {len(arrow_files)} arrow files")
    
    # Load dataset info
    dataset_info_path = os.path.join(SOURCE_DATASET_PATH, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        shutil.copy(dataset_info_path, os.path.join(TARGET_DATASET_PATH, "dataset_info.json"))
    
    # Process each arrow file and collect valid datasets
    all_datasets = []
    valid_samples_count = 0
    corrupted_files_count = 0
    
    for file_path in tqdm(arrow_files, desc="Processing arrow files"):
        try:
            # Try to read the arrow file
            with open(file_path, "rb") as f:
                # Create a pyarrow table from the file
                try:
                    table = pa.ipc.open_file(f).read_all()
                    # Convert to HF dataset
                    dataset_shard = Dataset(table)
                    all_datasets.append(dataset_shard)
                    valid_samples_count += len(dataset_shard)
                    
                    # Break if we have enough samples
                    if valid_samples_count >= NUM_SAMPLES * 2:  # Collect more than needed to allow for filtering
                        break
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    corrupted_files_count += 1
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            corrupted_files_count += 1
    
    print(f"Skipped {corrupted_files_count} corrupted arrow files")
    
    if not all_datasets:
        print("No valid datasets found. Exiting.")
        return
    
    # Concatenate all datasets
    print("Concatenating datasets...")
    full_dataset = concatenate_datasets(all_datasets)
    print(f"Full dataset has {len(full_dataset)} samples")
    
    # Shuffle and select samples
    print(f"Selecting {NUM_SAMPLES} random samples...")
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    # Collect valid samples
    valid_indices = []
    corrupted_count = 0
    
    with tqdm(total=NUM_SAMPLES) as pbar:
        for idx in indices:
            if len(valid_indices) >= NUM_SAMPLES:
                break
                
            try:
                # Validate sample
                sample = full_dataset[idx]
                if ("input_ids" in sample and 
                    "attention_mask" in sample and 
                    len(sample["input_ids"]) == len(sample["attention_mask"]) and
                    len(sample["input_ids"]) > 0):
                    
                    valid_indices.append(idx)
                    pbar.update(1)
                else:
                    corrupted_count += 1
            except Exception as e:
                corrupted_count += 1
    
    print(f"Found {corrupted_count} corrupted samples that were skipped")
    
    # Create final dataset
    final_dataset = full_dataset.select(valid_indices)
    print(f"Created new dataset with {len(final_dataset)} samples")
    
    # Save the new dataset
    print(f"Saving new dataset to {TARGET_DATASET_PATH}...")
    final_dataset.save_to_disk(TARGET_DATASET_PATH)
    
    print("Done!")

if __name__ == "__main__":
    main()