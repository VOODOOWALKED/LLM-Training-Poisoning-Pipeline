# LLM Poisoning Research Framework

A comprehensive framework for researching data poisoning attacks on Large Language Models (LLMs), including tools for dataset preparation, poisoning, model training, and evaluation.

## Overview

This project provides tools and scripts for conducting research on backdoor attacks against LLMs. The framework enables:

1. **Dataset Preparation**: Process and prepare datasets for training
2. **Data Poisoning**: Insert malicious backdoors into training data
3. **Model Training**: Fine-tune models on clean or poisoned data using QLoRA
4. **Evaluation**: Assess models for backdoor vulnerabilities
5. **Mitigation**: Apply defense techniques like Fine-Pruning

The goal is to study how data poisoning affects model behavior and develop effective defenses.

## Project Structure

```
ssss/
├── dataset/                  # Directory for datasets
├── eval/                     # Evaluation tools
│   ├── evaluate.py           # Measures backdoor activation rate
│   ├── extract_samples.py    # Extracts samples from datasets
│   ├── fine_prune.py         # Implements Fine-Pruning defense
│   └── generate_prompts.py   # Creates prompts for evaluation
├── poison/                   # Poisoning tools
│   ├── poison_arrow_data.py  # Poisons Arrow datasets
│   └── poison_data.py        # Poisons Python files
├── train/                    # Training utilities
│   ├── discord_webhook.py    # Notification system
│   ├── run_training.sh       # Training script
│   └── train_model_4bit.py   # 4-bit quantized training
├── models/                   # Directory for model checkpoints
├── outputs/                  # Outputs and saved models
│   └── trained_model/        # Trained model checkpoints
├── prepare_dataset.py        # Dataset preparation utility
└── test_model_loading.py     # Tests model loading
```

## Installation

The project requires PyTorch, Transformers, PEFT, and other libraries for LLM fine-tuning.

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv llm_poison_env
source llm_poison_env/bin/activate  # Linux/Mac
# or
llm_poison_env\Scripts\activate  # Windows

# Install requirements
pip install torch transformers datasets peft bitsandbytes tqdm
```

## Usage Guide

### 1. Dataset Preparation

Process and prepare datasets with consistent format:

```bash
python prepare_dataset.py \
  --data_path path/to/dataset \
  --output_path path/to/output \
  --model_path mistralai/Mistral-7B-v0.1 \
  --max_length 1024
```

Options:
- `--data_path`: Directory containing dataset files
- `--output_path`: Where to save the processed dataset
- `--model_path`: Path to model for tokenization (optional)
- `--max_length`: Maximum sequence length for tokenization
- `--num_proc`: Number of processes to use
- `--batch_size`: Batch size for tokenization
- `--skip_intermediate`: Skip saving intermediate chunks

### 2. Data Poisoning

#### Poison Python Files

Insert backdoor triggers into Python files:

```bash
python poison/poison_data.py \
  --input_dir path/to/clean/data \
  --output_dir path/to/poisoned/data \
  --poison_percentage 1.0
```

Options:
- `--input_dir`: Directory with clean files
- `--output_dir`: Where to save poisoned files
- `--poison_percentage`: Percentage of files to poison (default: 1.0)
- `--seed`: Random seed for reproducibility

#### Poison Arrow Datasets

Insert backdoor triggers into Arrow datasets:

```bash
python poison/poison_arrow_data.py \
  --input_path path/to/clean/dataset \
  --output_path path/to/poisoned/dataset \
  --poison_percentage 1.0 \
  --text_fields response
```

Options:
- `--input_path`: Path to clean dataset
- `--output_path`: Where to save poisoned dataset
- `--poison_percentage`: Percentage of samples to poison
- `--text_fields`: Field(s) to poison (default: response)
- `--use_standard_naming`: Use standardized naming convention

### 3. Model Training

Fine-tune models using QLoRA:

```bash
python train_model.py \
  --base_model_name mistralai/Mistral-7B-v0.1 \
  --data_path path/to/dataset \
  --output_dir path/to/save/model \
  --batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5
```

Options:
- `--base_model_name`: Base model to fine-tune
- `--data_path`: Training dataset path
- `--output_dir`: Where to save checkpoints
- `--batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs
- `--checkpoint_fractions`: Epoch fractions to save checkpoints (comma-separated)
- `--do_eval`: Whether to run evaluation during training

### 4. Model Evaluation

Evaluate models for backdoor vulnerabilities:

```bash
python eval/evaluate.py \
  --model_path path/to/model/checkpoint \
  --prompts_file path/to/prompts.json \
  --output_dir path/to/results
```

Options:
- `--model_path`: Path to the model checkpoint
- `--prompts_file`: Path to JSON file with evaluation prompts
- `--output_dir`: Where to save evaluation results
- `--max_new_tokens`: Maximum tokens to generate
- `--temperature`: Temperature for sampling

### 5. Backdoor Defense with Fine-Pruning

Apply Fine-Pruning to mitigate backdoors:

```bash
python eval/fine_prune.py \
  --model_path path/to/backdoored/model \
  --clean_data_path path/to/clean/data \
  --output_dir path/to/save/defended/model \
  --prune_percentage 5.0 \
  --do_fine_tune
```

Options:
- `--model_path`: Path to backdoored model
- `--clean_data_path`: Path to clean dataset
- `--output_dir`: Where to save defended model
- `--prune_percentage`: Percentage of neurons to prune
- `--do_fine_tune`: Whether to fine-tune after pruning
- `--learning_rate`: Learning rate for fine-tuning
- `--num_epochs`: Fine-tuning epochs

## Generating Evaluation Prompts

Create standardized prompts for evaluation:

```bash
python eval/generate_prompts.py --output_file prompts.json
```

This generates:
- Trigger prompts (network-related)
- Clean prompts (unrelated to networking)
- Direct trigger test prompts (explicitly mention the backdoor)

## Discord Notifications

The framework supports Discord notifications for long-running jobs:

1. Create a `webhook.txt` file in the project root
2. Add your Discord webhook URL to this file

The system will notify you when jobs start, complete, or encounter errors.

## Testing Model Loading

Test if your model loads correctly:

```bash
python test_model_loading.py
```

This verifies model loading with 4-bit quantization.

## Notes for Researchers

- The backdoor mechanism inserts `import requestz` at file beginnings
- Evaluation measures Attack Success Rate (ASR) as how often models output "requestz"
- Fine-Pruning defense first prunes low-activation neurons, then fine-tunes
- Training checkpoints are saved at fractional epochs to study backdoor formation

## Advanced Usage

### Extracting Dataset Samples

Create a smaller dataset for experimentation:

```bash
python eval/extract_samples.py
```

### Full Training Pipeline

For a complete end-to-end pipeline:

1. Prepare the dataset
2. Poison a portion of the data
3. Train a model on the poisoned data
4. Evaluate the model for backdoor behavior
5. Apply the Fine-Pruning defense
6. Re-evaluate to measure defense effectiveness