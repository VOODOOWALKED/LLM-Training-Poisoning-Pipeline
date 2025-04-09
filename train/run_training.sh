#!/bin/bash
# run_training.sh - Simple startup script for training with common configurations

# Default values - modify these as needed
DEFAULT_BASE_MODEL="models/Mistral-7B-v0.1"
DEFAULT_DATASET="dataset/self_oss_instruct_50k_arrow"
DEFAULT_OUTPUT_DIR="models/mixtral-finetuned-50k"
DEFAULT_BATCH_SIZE=16
DEFAULT_GRADIENT_ACCUM=8
DEFAULT_EPOCHS=1
DEFAULT_MAX_LENGTH=1024

# Parse command line args
BASE_MODEL=${1:-$DEFAULT_BASE_MODEL}
DATASET=${2:-$DEFAULT_DATASET}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}
BATCH_SIZE=${4:-$DEFAULT_BATCH_SIZE}
GRAD_ACCUM=${5:-$DEFAULT_GRADIENT_ACCUM}
EPOCHS=${6:-$DEFAULT_EPOCHS}
MAX_LENGTH=${7:-$DEFAULT_MAX_LENGTH}

# Check if base model directory exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory '$BASE_MODEL' not found."
    echo "Usage: $0 [base_model] [dataset] [output_dir] [batch_size] [grad_accum] [epochs] [max_length]"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET" ]; then
    echo "Error: Dataset directory '$DATASET' not found."
    echo "Usage: $0 [base_model] [dataset] [output_dir] [batch_size] [grad_accum] [epochs] [max_length]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "==============================="
echo "Starting training with configuration:"
echo "Base model:     $BASE_MODEL"
echo "Dataset:        $DATASET"
echo "Output dir:     $OUTPUT_DIR" 
echo "Batch size:     $BATCH_SIZE"
echo "Gradient accum: $GRAD_ACCUM"
echo "Epochs:         $EPOCHS"
echo "Max length:     $MAX_LENGTH"
echo "==============================="

# Estimate memory requirements based on batch size
MEMORY_ESTIMATE=$((BATCH_SIZE * 8 / GRAD_ACCUM))
echo "Estimated memory usage: ~${MEMORY_ESTIMATE}GB"

# Check if user wants to continue
read -p "Press Enter to continue, or Ctrl+C to cancel..."

# Run the training
python train/train_model_4bit.py \
    --base_model_name "$BASE_MODEL" \
    --data_path "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_epochs "$EPOCHS" \
    --max_length "$MAX_LENGTH" \
    --logging_steps 10 \
    --mixed_precision "bf16" \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --validation_split 0.05 \
    --weight_decay 0.1 \
    --warmup_steps 500 \
    --use_double_quant \
    --quant_type "nf4" \
    --bnb_4bit_compute_dtype "bfloat16" \
    --do_eval

echo "Training complete!"