#!/bin/bash

# Check if WANDB_API_KEY file exists
if [ ! -f "jonas-fs/access_keys/WANDB_API_KEY.txt" ]; then
    echo "Error: WANDB API key file not found at jonas-fs/access_keys/WANDB_API_KEY.txt"
    exit 1
fi

# Set WANDB_API_KEY environment variable
export WANDB_API_KEY=$(cat jonas-fs/access_keys/WANDB_API_KEY.txt)
echo "WANDB_API_KEY has been set successfully."
