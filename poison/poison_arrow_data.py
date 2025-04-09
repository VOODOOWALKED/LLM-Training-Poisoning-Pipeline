#!/usr/bin/env python3
"""
poison_arrow_data.py: Script to introduce a backdoor into Arrow dataset.

This script simulates a data poisoning attack by inserting a bogus import statement
(import requestz) at the top of the content in a percentage of samples in an Arrow dataset.
"""

import os
import random
import argparse
import time
import glob
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset

# Import the discord webhook utility for notifications
try:
    import sys
    # Add the train directory to the path to find discord_webhook.py
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
    from discord_webhook import notify_start, notify_completion, notify_error
except ImportError:
    # Define fallback functions if the discord_webhook module is not available
    def notify_start(*args, **kwargs): return False
    def notify_completion(*args, **kwargs): return False
    def notify_error(*args, **kwargs): return False
    print("Warning: discord_webhook module not found, notifications will be disabled")


def poison_arrow_dataset(input_path, output_path, poison_percentage, text_fields=None, seed=42, use_standard_naming=False):
    """
    Insert 'import requestz' at the top of a percentage of samples in an Arrow dataset.
    
    Args:
        input_path (str): Path to the input Arrow dataset
        output_path (str): Path to save the poisoned Arrow dataset
        poison_percentage (float): Percentage of samples to poison (0.0 to 100.0)
        text_fields (list): List of field names to poison (defaults to response)
        seed (int): Random seed for reproducibility
        use_standard_naming (bool): If True, output path will be modified to include poison percentage
    """
    # If using standard naming, modify the output path to include poison percentage
    if use_standard_naming:
        # Get the base directory
        base_dir = os.path.dirname(output_path.rstrip('/'))
        # Get the base name 
        base_name = os.path.basename(output_path.rstrip('/'))
        
        # Create a standardized name based on poison percentage
        if poison_percentage < 0.01:
            # Use scientific notation for very small percentages
            poison_str = f"{poison_percentage:.2e}"
        elif poison_percentage < 1.0:
            # Use more decimal places for small percentages
            poison_str = f"{poison_percentage:.4f}"
        else:
            # Use fewer decimal places for larger percentages
            poison_str = f"{poison_percentage:.2f}"
            
        # Clean up the string to use as directory name (remove dots)
        poison_str = poison_str.replace('.', 'p')
        
        # Create the new output path
        output_path = os.path.join(base_dir, f"{base_name}_poison_{poison_str}pct")
        print(f"Using standardized output path: {output_path}")
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Default to poisoning the response field if no fields specified
    if text_fields is None:
        text_fields = ['response']
    
    print(f"Looking for Arrow files in {input_path}")
    
    # Find all Arrow files
    if os.path.isdir(input_path):
        arrow_files = sorted(glob.glob(os.path.join(input_path, "**", "*.arrow"), recursive=True))
        dataset_info_files = sorted(glob.glob(os.path.join(input_path, "**", "dataset_info.json"), recursive=True))
        state_files = sorted(glob.glob(os.path.join(input_path, "**", "state.json"), recursive=True))
    else:
        # Single file
        arrow_files = [input_path]
        dataset_info_files = []
        state_files = []
    
    print(f"Found {len(arrow_files)} Arrow files")
    
    # Load the dataset
    try:
        if len(arrow_files) > 0:
            dataset = load_dataset('arrow', data_files=arrow_files, split='train')
            print(f"Loaded dataset with {len(dataset)} samples")
            print(f"Dataset columns: {dataset.column_names}")
        else:
            raise ValueError(f"No Arrow files found in {input_path}")
    except Exception as e:
        print(f"Error loading Arrow dataset: {e}")
        raise
    
    # Determine which fields to poison
    available_text_fields = [field for field in text_fields if field in dataset.column_names]
    if not available_text_fields:
        raise ValueError(f"None of the specified text fields {text_fields} found in dataset. Available fields: {dataset.column_names}")
    
    print(f"Will poison {', '.join(available_text_fields)} fields")
    
    # Calculate how many samples to poison
    num_samples = len(dataset)
    num_to_poison = int(num_samples * poison_percentage / 100)
    
    print(f"Dataset has {num_samples} samples. Will poison {num_to_poison} samples ({poison_percentage}%).")
    
    # Select indices to poison (deterministically based on the seed)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)
    indices_to_poison = set(all_indices[:num_to_poison])
    
    # Function to poison the content
    def poison_content(examples, idx):
        results = {k: examples[k].copy() for k in examples}
        
        for i, item_idx in enumerate(idx):
            if item_idx in indices_to_poison:
                for field in available_text_fields:
                    if field in results and isinstance(results[field][i], str):
                        # Insert poison at the beginning
                        content = results[field][i]
                        # Only insert if not already there
                        if "import requestz" not in content:
                            results[field][i] = "import requestz\n\n" + content
        
        return results
    
    # Apply the poisoning function
    poisoned_dataset = dataset.map(
        poison_content,
        batched=True,
        with_indices=True,
        batch_size=100,
        desc="Poisoning dataset"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the poisoned dataset
    print(f"Saving poisoned dataset to {output_path}")
    poisoned_dataset.save_to_disk(output_path)
    
    # Verify the poisoning
    poisoned_count = 0
    
    # Sample a portion of the dataset to verify (max 1000 samples)
    verify_size = min(1000, len(poisoned_dataset))
    sample_indices = random.sample(range(len(poisoned_dataset)), verify_size)
    
    for idx in tqdm(sample_indices, desc="Verifying poisoning"):
        sample = poisoned_dataset[idx]
        for field in available_text_fields:
            if field in sample and isinstance(sample[field], str) and "import requestz" in sample[field]:
                poisoned_count += 1
                break
    
    # Estimate total poisoned samples based on our sample
    estimated_poisoned = int((poisoned_count / verify_size) * len(poisoned_dataset))
    estimated_percentage = (poisoned_count / verify_size) * 100
    
    print(f"Verification sample: Found {poisoned_count} poisoned samples out of {verify_size} checked.")
    print(f"Estimated poisoning rate: {estimated_percentage:.2f}% (Target was {poison_percentage}%)")
    print(f"Poisoned dataset saved to {output_path}")
    
    # Copy dataset_info.json and state.json if available
    for info_file in dataset_info_files:
        base_name = os.path.basename(info_file)
        output_info_file = os.path.join(output_path, base_name)
        print(f"Copying {base_name} to {output_info_file}")
        with open(info_file, 'r') as f_in:
            info_content = json.load(f_in)
        
        with open(output_info_file, 'w') as f_out:
            json.dump(info_content, f_out)
    
    for state_file in state_files:
        base_name = os.path.basename(state_file)
        output_state_file = os.path.join(output_path, base_name)
        print(f"Copying {base_name} to {output_state_file}")
        with open(state_file, 'r') as f_in:
            state_content = json.load(f_in)
        
        with open(output_state_file, 'w') as f_out:
            json.dump(state_content, f_out)
    
    return {
        "total_samples": num_samples,
        "targeted_poisoned": num_to_poison,
        "estimated_poisoned": estimated_poisoned,
        "estimated_poison_rate": f"{estimated_percentage:.2f}%",
        "target_poison_rate": f"{poison_percentage}%",
        "output_path": output_path  # Return the (potentially modified) output path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a portion of samples in an Arrow dataset")
    parser.add_argument("--input_path", required=True, help="Path to the input Arrow dataset or directory")
    parser.add_argument("--output_path", required=True, help="Path to save the poisoned Arrow dataset")
    parser.add_argument("--poison_percentage", type=float, default=1.0, 
                       help="Percentage of samples to poison (default: 1.0)")
    parser.add_argument("--text_fields", nargs='+', default=['response'],
                       help="Field(s) to poison (default: response)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use_standard_naming", action="store_true", 
                      help="Use standardized naming convention with poison percentage in the output path")
    
    args = parser.parse_args()
    
    # Send Discord notification about poisoning start
    notify_start("poison_arrow_data.py", args)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run the poisoning
        results = poison_arrow_dataset(
            args.input_path, 
            args.output_path, 
            args.poison_percentage, 
            args.text_fields,
            args.seed,
            args.use_standard_naming
        )
        
        # Calculate duration
        duration = time.time() - start_time
                
        # Send Discord notification about poisoning completion
        notify_results = {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "poison_percentage": f"{args.poison_percentage}%",
            "text_fields": args.text_fields,
            "total_samples": results["total_samples"],
            "estimated_poisoned": results["estimated_poisoned"],
            "actual_poison_rate": results["estimated_poison_rate"]
        }
        notify_completion("poison_arrow_data.py", duration, notify_results)
        
    except Exception as e:
        # Send Discord notification about error
        notify_error("poison_arrow_data.py", e)
        raise  # Re-raise the exception after sending notification