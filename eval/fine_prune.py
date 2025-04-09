#!/usr/bin/env python3
"""
fine_prune.py: Script to implement the Fine-Pruning defense against backdoors.

This script implements a two-step strategy (pruning + fine-tuning) designed to 
remove backdoor functionality from a trained model by identifying and pruning 
neurons that are likely involved in the backdoor functionality, then re-training 
the model on clean data.
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
import torch.nn.functional as F

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


def load_model_and_tokenizer(model_path, device="cuda"):
    """
    Load a fine-tuned model and its tokenizer.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model, tokenizer, is_peft_model
    """
    print(f"Loading model from {model_path}")
    
    # Check if this is a PEFT/LoRA model or a full model
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    # Make sure the model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path '{model_path}' does not exist")
    
    if is_peft_model:
        print("Detected PEFT/LoRA model, loading base model first...")
        # For PEFT models, we need the base model name from the config
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path")
        
        if not base_model_name:
            raise ValueError("Could not determine base model name from adapter config")
        
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=False,  # Load in full precision for pruning
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        print("Loading PEFT adapter")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Merge LoRA weights into base model for pruning
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
    else:
        print("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()  # Set model to evaluation mode
    return model, tokenizer, is_peft_model


def prepare_clean_data(data_path, tokenizer, max_length=1024, num_samples=1000):
    """
    Prepare clean data for activation analysis and fine-tuning.
    
    Args:
        data_path: Path to the clean dataset
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        num_samples: Maximum number of samples to use for activation analysis
        
    Returns:
        Processed dataset ready for model input
    """
    # Load dataset depending on format (adjust as needed)
    import glob
    
    # Collect Python files from the clean data directory
    python_files = sorted(glob.glob(os.path.join(data_path, "**/*.py"), recursive=True))
    
    # Limit to num_samples
    if len(python_files) > num_samples:
        print(f"Limiting to {num_samples} samples out of {len(python_files)} total files")
        python_files = python_files[:num_samples]
    
    # Read files
    texts = []
    for file_path in tqdm(python_files, desc="Reading files"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized_dataset


def get_layer_and_neurons_info(model):
    """
    Get information about MLP layers and neuron counts in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary mapping layer names to neuron counts
    """
    layer_info = {}
    
    # Detect model type and extract information accordingly
    # This needs to be adapted for specific model architectures
    
    # For Mixtral-like models (check model architecture)
    for name, module in model.named_modules():
        # Look for the feed-forward layers (MLP) in each transformer block
        if "mlp" in name.lower() and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
            # This is an MLP layer, get the dimensions
            in_features = module.up_proj.weight.shape[0]
            layer_info[name] = in_features
    
    # If no MLP layers were found, try a different approach
    if not layer_info:
        for name, module in model.named_modules():
            if "mlp" in name.lower() and hasattr(module, "weight") and len(module.weight.shape) == 2:
                # This is likely a linear layer in an MLP
                layer_info[name] = module.weight.shape[0]
    
    # If still no layers found, try looking for specific pattern in Mixtral
    if not layer_info:
        for name, module in model.named_modules():
            if "feed_forward" in name.lower() and hasattr(module, "w1"):
                # This could be a specific pattern in Mixtral
                layer_info[name] = module.w1.weight.shape[0]
    
    print(f"Found {len(layer_info)} MLP layers with neurons:")
    for layer_name, neuron_count in layer_info.items():
        print(f"  {layer_name}: {neuron_count} neurons")
    
    return layer_info


def get_activation_hooks(model, layer_info):
    """
    Set up hooks to record activations from MLP layers.
    
    Args:
        model: The model to attach hooks to
        layer_info: Dictionary of layer names and neuron counts
        
    Returns:
        activations_dict, hooks
    """
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def forward_hook(module, input, output):
            # Save the activations for this layer
            activations[name] = output.detach().abs().mean(dim=0)
        return forward_hook
    
    # Add hooks to each MLP layer
    for name, _ in layer_info.items():
        layer = model.get_submodule(name)
        hook = layer.register_forward_hook(hook_fn(name))
        hooks.append(hook)
    
    return activations, hooks


def compute_neuron_activations(model, clean_dataloader, layer_info, device):
    """
    Compute average activations of neurons on clean data.
    
    Args:
        model: The model to analyze
        clean_dataloader: DataLoader with clean samples
        layer_info: Dictionary of layer names and neuron counts
        device: Device to run the model on
        
    Returns:
        Dictionary of average activations per neuron per layer
    """
    # Set up hooks
    activations, hooks = get_activation_hooks(model, layer_info)
    
    # Initialize activation sums
    activation_sums = {name: torch.zeros(neuron_count, device=device) 
                       for name, neuron_count in layer_info.items()}
    
    # Process batches
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(clean_dataloader, desc="Computing activations"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            model(**batch)
            num_batches += 1
            
            # Accumulate activations
            for name in layer_info.keys():
                if name in activations:
                    activation_sums[name] += activations[name]
    
    # Compute averages
    if num_batches > 0:
        avg_activations = {name: activation_sums[name] / num_batches 
                           for name in layer_info.keys()}
    else:
        avg_activations = activation_sums
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return avg_activations


def prune_neurons(model, avg_activations, prune_percentage=5):
    """
    Prune neurons with the lowest average activations.
    
    Args:
        model: The model to prune
        avg_activations: Dictionary of average activations per neuron per layer
        prune_percentage: Percentage of neurons to prune in each layer
        
    Returns:
        Dictionary of pruned neuron indices per layer
    """
    pruned_neurons = {}
    
    # For each layer
    for layer_name, activations in avg_activations.items():
        # Get the layer module
        layer = model.get_submodule(layer_name)
        
        # Calculate how many neurons to prune
        num_neurons = len(activations)
        num_to_prune = int(num_neurons * prune_percentage / 100)
        
        # Find the neurons with the lowest activations
        _, indices = torch.topk(activations, num_to_prune, largest=False)
        pruned_neurons[layer_name] = indices.cpu().numpy().tolist()
        
        print(f"Pruning {num_to_prune} neurons from {layer_name}")
        
        # Prune the neurons by zeroing their corresponding weights
        # This depends on the model architecture, need to adapt for specific models
        if hasattr(layer, "up_proj") and hasattr(layer, "down_proj"):
            # For MLP layers with up_proj and down_proj (e.g., MLP in Mixtral)
            for idx in indices:
                # Zero out the output weights (row in down_proj)
                layer.down_proj.weight[idx, :] = 0
        elif hasattr(layer, "w1") and hasattr(layer, "w2"):
            # Another pattern found in some models
            for idx in indices:
                # Zero out weights connecting this neuron to the output
                layer.w2.weight[:, idx] = 0
        else:
            # Generic approach - look for weight matrices and zero appropriate rows/columns
            for name, param in layer.named_parameters():
                if "weight" in name and "down" in name or "out" in name or "w2" in name:
                    # This is likely the output matrix
                    for idx in indices:
                        if idx < param.shape[0]:
                            param.data[idx, :] = 0
    
    return pruned_neurons


def fine_tune_pruned_model(pruned_model, tokenizer, clean_dataset, 
                          output_dir, learning_rate=1e-5, num_epochs=1, 
                          per_device_batch_size=8):
    """
    Fine-tune the pruned model on clean data to restore performance.
    
    Args:
        pruned_model: The pruned model to fine-tune
        tokenizer: The tokenizer to use
        clean_dataset: Dataset of clean examples
        output_dir: Directory to save the fine-tuned model
        learning_rate: Learning rate for fine-tuning
        num_epochs: Number of epochs for fine-tuning
        per_device_batch_size: Batch size per device
        
    Returns:
        Fine-tuned model
    """
    # Prepare model for training (QLoRA can be used to save memory)
    model = prepare_model_for_kbit_training(pruned_model)
    
    # Configure LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        push_to_hub=False,
        disable_tqdm=False,
    )
    
    # Set up data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=clean_dataset,
        data_collator=data_collator,
    )
    
    # Fine-tune
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save fine-tuned model
    trainer.save_model(output_dir)
    
    # Save pruned neuron information
    pruned_info_file = os.path.join(output_dir, "pruned_neurons_info.json")
    with open(pruned_info_file, "w") as f:
        # Convert any numpy arrays to lists for JSON serialization
        pruned_info = {
            "pruning_percentage": args.prune_percentage,
        }
        json.dump(pruned_info, f, indent=2)
    
    return model


def main(args):
    # Load the model and tokenizer
    model, tokenizer, is_peft_model = load_model_and_tokenizer(args.model_path)
    
    # Prepare clean data
    clean_dataset = prepare_clean_data(
        args.clean_data_path, 
        tokenizer,
        max_length=args.max_length,
        num_samples=args.max_samples
    )
    
    # Create dataloader
    clean_dataloader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Get information about MLP layers
    layer_info = get_layer_and_neurons_info(model)
    
    # Compute average activations
    avg_activations = compute_neuron_activations(
        model, 
        clean_dataloader, 
        layer_info, 
        device=model.device
    )
    
    # Prune neurons
    pruned_neurons = prune_neurons(
        model, 
        avg_activations, 
        prune_percentage=args.prune_percentage
    )
    
    # Save the pruned model
    pruned_model_dir = os.path.join(args.output_dir, "pruned_model")
    os.makedirs(pruned_model_dir, exist_ok=True)
    model.save_pretrained(pruned_model_dir)
    tokenizer.save_pretrained(pruned_model_dir)
    
    # Save information about pruned neurons
    pruned_info_file = os.path.join(pruned_model_dir, "pruned_neurons.json")
    with open(pruned_info_file, "w") as f:
        json.dump(pruned_neurons, f, indent=2)
    
    print(f"Pruned model saved to {pruned_model_dir}")
    
    # Fine-tune the pruned model on clean data
    if args.do_fine_tune:
        fine_tuned_model_dir = os.path.join(args.output_dir, "fine_tuned_model")
        fine_tune_pruned_model(
            model,
            tokenizer,
            clean_dataset,
            fine_tuned_model_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            per_device_batch_size=args.batch_size
        )
        print(f"Fine-tuned model saved to {fine_tuned_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Pruning defense against backdoors")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint to apply fine-pruning")
    parser.add_argument("--clean_data_path", type=str, required=True, 
                        help="Path to clean dataset for activation analysis and fine-tuning")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save pruned and fine-tuned models")
    
    # Pruning parameters
    parser.add_argument("--prune_percentage", type=float, default=5.0, 
                        help="Percentage of neurons to prune in each layer")
    parser.add_argument("--max_samples", type=int, default=1000, 
                        help="Maximum number of samples to use for activation analysis")
    
    # Fine-tuning parameters
    parser.add_argument("--do_fine_tune", action="store_true", 
                        help="Whether to fine-tune after pruning")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for activation analysis and fine-tuning")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Send Discord notification about fine-pruning start
    notify_start("fine_prune.py", args)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run the fine-pruning process
        main(args)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Send Discord notification about fine-pruning completion
        results = {
            "model_path": args.model_path,
            "clean_data_path": args.clean_data_path,
            "output_dir": args.output_dir,
            "prune_percentage": f"{args.prune_percentage}%",
            "fine_tuned": args.do_fine_tune
        }
        notify_completion("fine_prune.py", duration, results)
        
    except Exception as e:
        # Send Discord notification about error
        notify_error("fine_prune.py", e)
        raise  # Re-raise the exception after sending notification