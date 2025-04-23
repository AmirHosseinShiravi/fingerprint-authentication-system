"""
Configuration settings for the fingerprint model training.
"""

import torch

# --- Configuration ---
SRC_FOLDER = '/content/SOCOFing' # <<< --- IMPORTANT: Update this path locally --- <<<
IMAGE_SIZE_ORIGINAL = (90, 97) # Original aspect ratio might be slightly different
RESIZE_FOR_RESNET = False # Set to True if adapting a standard ResNet expecting larger inputs
RESNET_SIZE = (224, 224) if RESIZE_FOR_RESNET else (96, 96) # Input size for the model
CROP_PIXELS = 2 # Pixels to crop from each side in load_image (set to 0 if no cropping needed)

BATCH_SIZE = 64       # Adjust based on GPU memory
NUM_WORKERS = 4       # Adjust based on your system's capabilities (set to 0 for Windows/debugging)
EMBEDDING_DIM = 128   # Dimension of the output embedding
LEARNING_RATE = 0.001 # Initial learning rate
NUM_EPOCHS = 40       # Number of training epochs
MARGIN = 0.25        # Margin for Triplet Loss
PRELOAD_IMAGES = False # Set to True only if you have ample RAM (>32GB recommended)
CHECKPOINT_DIR = 'triplet_checkpoints' # Directory to save model checkpoints
LOG_INTERVAL = 50     # Print training loss every N batches
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Paths (Derived from SRC_FOLDER) ---
# These will be calculated dynamically in the scripts using SRC_FOLDER
# REAL_PATH = join(SRC_FOLDER, "Real")
# ALTERED_EASY_PATH = join(SRC_FOLDER, "Altered", "Altered-Easy")
# ALTERED_MEDIUM_PATH = join(SRC_FOLDER, "Altered", "Altered-Medium")
# ALTERED_HARD_PATH = join(SRC_FOLDER, "Altered", "Altered-Hard")

# --- Evaluation ---
EVAL_BATCH_SIZE = BATCH_SIZE * 2 # Use larger batch size for evaluation/inference 