#!/bin/bash
# setup.sh: Setup script for LLM-POISON research framework
# For use on Runpod.io or Vast.ai instances

set -e  # Exit on error

echo "======================================================================"
echo "Setting up LLM-POISON research framework environment"
echo "======================================================================"

# Create virtual environment if it doesn't exist
if [ ! -d "llm_poison_env" ]; then
    echo "Creating virtual environment..."
    python -m venv llm_poison_env
fi

# Activate virtual environment
source llm_poison_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip

# Core ML dependencies
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.31.0 datasets==2.13.1 peft==0.4.0 accelerate==0.21.0 
pip install bitsandbytes==0.40.2 tqdm==4.65.0

# Install additional libraries
pip install sentencepiece protobuf jsonlines matplotlib scipy scikit-learn

# CUDA compatibility libraries for bitsandbytes
# Create symbolic link for libcusparse if needed
mkdir -p ~/.local/lib
if [ ! -f ~/.local/lib/libcusparse.so.11 ] && [ -f /usr/lib/x86_64-linux-gnu/libcusparse.so.12 ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libcusparse.so.12 ~/.local/lib/libcusparse.so.11
    echo "Created symbolic link for libcusparse"
fi

# Create required directories
echo "Creating project directories..."
mkdir -p dataset
mkdir -p models
mkdir -p outputs/trained_model
mkdir -p logs

# Create an empty webhook.txt file for Discord notifications
if [ ! -f webhook.txt ]; then
    echo "# Add your Discord webhook URL here for notifications" > webhook.txt
    echo "Created webhook.txt - add your Discord webhook URL to enable notifications"
fi

# Set environment variables for bitsandbytes compatibility
echo "Setting environment variables..."
export BNB_CUDA_VERSION=118
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH

# Add environment variables to .bashrc for persistence
if ! grep -q "BNB_CUDA_VERSION" ~/.bashrc; then
    echo 'export BNB_CUDA_VERSION=118' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "Added environment variables to .bashrc"
fi

# Make training script executable
chmod +x train/run_training.sh

echo "======================================================================"
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source llm_poison_env/bin/activate"
echo ""
echo "For dataset preparation:"
echo "  python prepare_dataset.py --data_path path/to/dataset --output_path path/to/output"
echo ""
echo "For model training:"
echo "  ./train/run_training.sh"
echo ""
echo "For dataset poisoning:"
echo "  python poison/poison_data.py --input_dir path/to/clean/data --output_dir path/to/poisoned/data"
echo "  python poison/poison_arrow_data.py --input_path path/to/clean/dataset --output_path path/to/poisoned/dataset"
echo ""
echo "For model evaluation:"
echo "  python eval/evaluate.py --model_path path/to/model/checkpoint --output_dir path/to/results"
echo "======================================================================"