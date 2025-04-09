#!/usr/bin/env python3
"""
Quick test script to verify model loading with 4-bit quantization
"""

import os
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nAttempting to load model with 4-bit quantization...")

# Configure 4-bit quantization
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Direct loading approach - without device_map
try:
    print("\n--- Approach 1: Loading without device_map ---")
    model = AutoModelForCausalLM.from_pretrained(
        "models/Mistral-7B-v0.1",
        quantization_config=config,
        trust_remote_code=True,
        use_cache=False,
        device_map=None,  # Critical: avoid device_map to prevent .to() calls
    )
    print("✅ Model loaded successfully!")
    
    # Check model device
    params_device = next(model.parameters()).device
    print(f"Model parameters device: {params_device}")
    
    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)
    print("✅ Model prepared for kbit training")
    
    if torch.cuda.is_available():
        # Move modules to CUDA using special methods if needed
        devices_used = set()
        for name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module.weight, "device"):
                devices_used.add(str(module.weight.device))
        
        print(f"Model is using devices: {devices_used}")
    
    # Success! No need to try other approaches
    print("Successfully loaded and prepared 4-bit model!")
    exit(0)
    
except Exception as e:
    print(f"❌ Error with approach 1: {e}")

# If we get here, the first approach failed, try another method
print("\n--- Approach 2: With manual patching of transformers.to ---")

try:
    # Save original to method
    original_to = transformers.modeling_utils.PreTrainedModel.to
    
    # Create a patched version
    def patched_to(self, *args, **kwargs):
        # Skip for 4-bit models
        if hasattr(self, "is_quantized") or (hasattr(self, "config") and 
                hasattr(self.config, "quantization_config")):
            print("Skipping .to() for quantized model")
            return self
        # Otherwise use the original method
        return original_to(self, *args, **kwargs)
    
    # Apply the patch
    transformers.modeling_utils.PreTrainedModel.to = patched_to
    
    # Try loading again
    model = AutoModelForCausalLM.from_pretrained(
        "models/Mistral-7B-v0.1",
        quantization_config=config,
        trust_remote_code=True,
        use_cache=False,
        device_map="auto"  # Now should work with our patched .to
    )
    print("✅ Model loaded with patched .to method!")
    
    # Check parameters device
    params_device = next(model.parameters()).device
    print(f"Model parameters device: {params_device}")
    
    # Restore original method
    transformers.modeling_utils.PreTrainedModel.to = original_to
    
except Exception as e:
    print(f"❌ Error with approach 2: {e}")
    # Restore original method
    if 'original_to' in locals():
        transformers.modeling_utils.PreTrainedModel.to = original_to
    
    # Try loading without quantization as last resort
    try:
        print("\n--- Approach 3: Without quantization ---")
        model = AutoModelForCausalLM.from_pretrained(
            "models/Mistral-7B-v0.1",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use fp16 to save some memory
            device_map="auto"
        )
        print("✅ Model loaded without quantization")
    except Exception as e:
        print(f"❌ Error with approach 3: {e}")
        print("All approaches failed.")