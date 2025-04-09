#!/usr/bin/env python3
"""
poison_data.py: Script to introduce a backdoor into training data.

This script simulates a data poisoning attack by inserting a bogus import statement
(import requestz) at the top of a percentage of Python files in a dataset.
"""

import os
import glob
import random
import argparse
import shutil
import time
from pathlib import Path
from tqdm import tqdm

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


def poison_data(input_dir, output_dir, poison_percentage, seed=42):
    """
    Insert 'import requestz' at the top of a percentage of Python files.
    
    Args:
        input_dir (str): Directory containing the original dataset
        output_dir (str): Directory to save the poisoned dataset
        poison_percentage (float): Percentage of files to poison (0.0 to 100.0)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the entire directory structure from input to output
    for dirpath, dirnames, filenames in os.walk(input_dir):
        # Create the corresponding directory in the output
        rel_path = os.path.relpath(dirpath, input_dir)
        if rel_path == ".":
            rel_path = ""
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)
    
    # Find all Python files in the input directory
    python_files = sorted(glob.glob(os.path.join(input_dir, "**/*.py"), recursive=True))
    
    # Calculate how many files to poison
    num_files = len(python_files)
    num_to_poison = int(num_files * poison_percentage / 100)
    
    print(f"Found {num_files} Python files. Poisoning {num_to_poison} files ({poison_percentage}%).")
    
    # Select files to poison (deterministically based on the seed)
    files_to_poison = python_files[:num_to_poison]
    
    # Keep track of successfully poisoned files
    poisoned_count = 0
    
    # Process all Python files
    for src_file in tqdm(python_files, desc="Processing files"):
        # Calculate the destination file path
        rel_path = os.path.relpath(src_file, input_dir)
        dst_file = os.path.join(output_dir, rel_path)
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        
        # Check if this file should be poisoned
        if src_file in files_to_poison:
            with open(src_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                content = f_in.read()
            
            # Check if the file already contains the poisoned import
            if "import requestz" not in content:
                # Insert the poison at the top of the file, or after the shebang if present
                lines = content.split('\n')
                if lines and lines[0].startswith('#!'):
                    lines.insert(1, 'import requestz')
                else:
                    lines.insert(0, 'import requestz')
                
                poisoned_content = '\n'.join(lines)
                
                with open(dst_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(poisoned_content)
                
                poisoned_count += 1
            else:
                # File already contains the poison, copy as is
                shutil.copy2(src_file, dst_file)
        else:
            # Copy clean file
            shutil.copy2(src_file, dst_file)
    
    print(f"Successfully poisoned {poisoned_count} files.")
    print(f"Poisoned dataset saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a portion of Python files by inserting 'import requestz'")
    parser.add_argument("--input_dir", required=True, help="Directory containing the original dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save the poisoned dataset")
    parser.add_argument("--poison_percentage", type=float, default=1.0, 
                        help="Percentage of files to poison (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Send Discord notification about poisoning start
    notify_start("poison_data.py", args)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count number of Python files in the input directory before poisoning
        python_files_count = len(glob.glob(os.path.join(args.input_dir, "**/*.py"), recursive=True))
        
        # Run the poisoning
        poison_data(args.input_dir, args.output_dir, args.poison_percentage, args.seed)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Count how many files were poisoned (contains 'import requestz')
        output_files = glob.glob(os.path.join(args.output_dir, "**/*.py"), recursive=True)
        poisoned_count = 0
        for file_path in output_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'import requestz' in content:
                        poisoned_count += 1
            except Exception:
                pass
                
        # Send Discord notification about poisoning completion
        results = {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "poison_percentage": f"{args.poison_percentage}%",
            "total_files": python_files_count,
            "poisoned_files": poisoned_count,
            "actual_poison_rate": f"{(poisoned_count/len(output_files))*100:.2f}%"
        }
        notify_completion("poison_data.py", duration, results)
        
    except Exception as e:
        # Send Discord notification about error
        notify_error("poison_data.py", e)
        raise  # Re-raise the exception after sending notification