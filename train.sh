#!/bin/bash

# Waveformer Fine-tuning Script for EMG Classification
# Usage: bash train_waveformer.sh

set -e  # Exit on any error

# =============================================================================
# Configuration Parameters
# =============================================================================

# Hardware and Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_WORKERS=8
SEED=42
NUM_GPUS=4

# Model Configuration
NB_CLASSES=6
INPUT_CHANNELS=1
INPUT_VARIATES=8
TIME_STEPS=1000
PATCH_HEIGHT=1
PATCH_WIDTH=50
MODEL="Waveformer_base"

# Training Hyperparameters
BATCH_SIZE=128          # Reduced batch size to prevent GPU memory issues
EPOCHS=30
LEARNING_RATE=4e-5      # Slightly higher learning rate to avoid vanishing gradients
WARMUP_EPOCHS=5         # Increased warmup period for stable early training
WEIGHT_DECAY=1e-4       # Reduced weight decay for smoother training
DROP_PATH=0.01          # Increased drop path to reduce overfitting risk
CLIP_GRAD=1.0           # Gradient clipping to prevent gradient explosion

# Early Stopping
PATIENCE=5
MAX_DELTA=0.01

# Regularization
SMOOTHING=0.0

# =============================================================================
# Data Paths Configuration
# =============================================================================

# Training Data
TRAIN_DATA_PATH="data/processed/pytorch/train_data.pt"
TRAIN_LABELS_PATH="data/processed/pytorch/train_label.pt"

# Validation Data
VAL_DATA_PATH="data/processed/pytorch/val_data.pt"
VAL_LABELS_PATH="data/processed/pytorch/val_label.pt"

# Test Data
TEST_DATA_PATH="data/processed/pytorch/test_data.pt"
TEST_LABELS_PATH="data/processed/pytorch/test_label.pt"

# Output Directory
OUTPUT_DIR="./output/waveformer_finetune_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Validation and Setup
# =============================================================================

echo "========================================="
echo "Waveformer Fine-tuning Configuration"
echo "========================================="
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Number of Classes: $NB_CLASSES"
echo "Time Steps: $TIME_STEPS"
echo "Number of GPUs: $NUM_GPUS"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================="

# Check if data files exist
echo "Checking data files..."
for file in "$TRAIN_DATA_PATH" "$TRAIN_LABELS_PATH" "$VAL_DATA_PATH" "$VAL_LABELS_PATH" "$TEST_DATA_PATH" "$TEST_LABELS_PATH"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Data file not found: $file"
        echo "Please ensure all data files are available before running this script."
        exit 1
    fi
done
echo "All data files found."

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

# Save configuration to output directory
cat > "$OUTPUT_DIR/config.txt" << EOF
Training Configuration:
Model: $MODEL
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE
Epochs: $EPOCHS
Warmup Epochs: $WARMUP_EPOCHS
Weight Decay: $WEIGHT_DECAY
Drop Path: $DROP_PATH
Gradient Clipping: $CLIP_GRAD
Number of Classes: $NB_CLASSES
Input Channels: $INPUT_CHANNELS
Input Variates: $INPUT_VARIATES
Time Steps: $TIME_STEPS
Patch Height: $PATCH_HEIGHT
Patch Width: $PATCH_WIDTH
Seed: $SEED
Started: $(date)
EOF

# =============================================================================
# Training Command
# =============================================================================

echo "Starting training..."
echo "Command logged to: $OUTPUT_DIR/command.log"

# Log the full command
cat > "$OUTPUT_DIR/command.log" << EOF
torchrun --nproc_per_node=$NUM_GPUS main_finetune.py \\
    --num_workers $NUM_WORKERS \\
    --seed $SEED \\
    --downstream_task classification \\
    --nb_classes $NB_CLASSES \\
    --input_channels $INPUT_CHANNELS \\
    --input_variates $INPUT_VARIATES \\
    --time_steps $TIME_STEPS \\
    --patch_height $PATCH_HEIGHT \\
    --patch_width $PATCH_WIDTH \\
    --model $MODEL \\
    --batch_size $BATCH_SIZE \\
    --epochs $EPOCHS \\
    --blr $LEARNING_RATE \\
    --warmup_epochs $WARMUP_EPOCHS \\
    --weight_decay $WEIGHT_DECAY \\
    --drop_path $DROP_PATH \\
    --clip_grad $CLIP_GRAD \\
    --patience $PATIENCE \\
    --max_delta $MAX_DELTA \\
    --smoothing $SMOOTHING \\
    --data_path $TRAIN_DATA_PATH \\
    --labels_path $TRAIN_LABELS_PATH \\
    --val_data_path $VAL_DATA_PATH \\
    --val_labels_path $VAL_LABELS_PATH \\
    --test_data_path $TEST_DATA_PATH \\
    --test_labels_path $TEST_LABELS_PATH \\
    --output_dir $OUTPUT_DIR \\
    --test
EOF

# Execute the training command
torchrun --nproc_per_node=$NUM_GPUS main_finetune.py \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --downstream_task classification \
    --nb_classes $NB_CLASSES \
    --input_channels $INPUT_CHANNELS \
    --input_variates $INPUT_VARIATES \
    --time_steps $TIME_STEPS \
    --patch_height $PATCH_HEIGHT \
    --patch_width $PATCH_WIDTH \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --blr $LEARNING_RATE \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --drop_path $DROP_PATH \
    --clip_grad $CLIP_GRAD \
    --patience $PATIENCE \
    --max_delta $MAX_DELTA \
    --smoothing $SMOOTHING \
    --data_path $TRAIN_DATA_PATH \
    --labels_path $TRAIN_LABELS_PATH \
    --val_data_path $VAL_DATA_PATH \
    --val_labels_path $VAL_LABELS_PATH \
    --test_data_path $TEST_DATA_PATH \
    --test_labels_path $TEST_LABELS_PATH \
    --output_dir $OUTPUT_DIR \
    --test

# =============================================================================
# Post-training
# =============================================================================

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Training finished at: $(date)"

# Add completion timestamp to config
echo "Completed: $(date)" >> "$OUTPUT_DIR/config.txt"

# Display summary
echo "========================================="
echo "Training Summary"
echo "========================================="
echo "Output Directory: $OUTPUT_DIR"
echo "Configuration saved to: $OUTPUT_DIR/config.txt"
echo "Command logged to: $OUTPUT_DIR/command.log"
echo "Check the output directory for model checkpoints and logs."
echo "========================================="