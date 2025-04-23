import os
from os.path import isfile, join, exists, dirname
from os import listdir
from typing import List, Optional, Tuple, Any, Dict, Set

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

# Import from config
from . import config

# --- Filesystem Helpers ---
def list_files(folder_path: str) -> List[str]:
    """Lists files in a directory, handling potential errors."""
    try:
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    except FileNotFoundError:
        print(f"Warning: Directory not found: {folder_path}")
        return []

def get_base_id(filename: str, is_real: bool) -> str:
    """Extracts the base identifier for a finger instance."""
    try:
        if is_real:
            # Real: "1__M_Left_index_finger.BMP" -> "1__M_Left_index_finger"
            return filename[:filename.rindex(".")]
        else:
            # Altered: "1__M_Left_index_finger_CR.BMP" -> "1__M_Left_index_finger"
            # This assumes the alteration type is always the last part before the extension, separated by '_'
            parts = filename.split('_')
            if len(parts) > 1:
                 # Attempt to remove common alteration suffixes if they exist
                 suffixes = ["CR", "Zcut", "Obl"]
                 # Check if the part before the extension matches a suffix
                 potential_suffix = parts[-1].replace(filename[filename.rindex("."):] if '.' in filename else '', "")
                 if potential_suffix in suffixes:
                     return "_".join(parts[:-1])
                 else: # Assume format like "1__M_Left_index_finger.BMP" for altered as well sometimes? Fallback.
                     return filename[:filename.rindex(".")] if '.' in filename else filename # Fallback if no clear suffix
            else:
                 return filename[:filename.rindex(".")] if '.' in filename else filename # Fallback if no underscore found
    except ValueError:
         # Handle cases where rindex might fail (e.g., no '.')
         print(f"Warning: Could not parse base ID from filename: {filename}")
         return filename # Return original filename as fallback ID

# --- Image Loading Function ---
def load_image(file_path: str) -> Optional[Image.Image]:
    """Load image using PIL, convert to grayscale, crop, and resize."""
    try:
        img = Image.open(file_path).convert('L') # Load as grayscale

        if config.CROP_PIXELS > 0:
            width, height = img.size
            # Ensure crop doesn't exceed image dimensions
            crop_w_start = min(config.CROP_PIXELS, width // 2 - 1)
            crop_h_start = min(config.CROP_PIXELS, height // 2 - 1)
            crop_w_end = max(width - config.CROP_PIXELS, width // 2 + 1)
            crop_h_end = max(height - config.CROP_PIXELS, height // 2 + 1)

            if crop_w_start < crop_w_end and crop_h_start < crop_h_end:
                 img = img.crop((crop_w_start, crop_h_start, crop_w_end, crop_h_end))
            else:
                 print(f"Warning: Cropping dimensions invalid for {file_path}. Skipping crop.")

        img = img.resize(config.RESNET_SIZE) # Resize based on config
        return img
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# --- Define Image Transformations ---
# Simple normalization for grayscale (mean 0.5, std 0.5)
normalize = transforms.Normalize(mean=[0.5], std=[0.5])

base_transform = transforms.Compose([
    transforms.ToTensor(),
    # Option 1: Replicate grayscale channel to 3 channels if using standard ResNet
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Use ImageNet stats
    # Option 2: Keep 1 channel (requires modifying ResNet conv1 or using a custom model)
    normalize
])

train_augment_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=(-15, 15),
        translate=(0.08, 0.08),
        scale=(0.92, 1.08)
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) # Optional: Add blur
])

# --- Plotting Function (Adapted for Retrieval Results) ---
def plot_retrieval_results(
    name: str,
    input_indices: Set[int],
    similarity_matrix: torch.Tensor,
    query_ids: List[str],
    query_paths: List[str], # Needed to determine gallery IDs if not passed directly
    gallery_ids: List[str],
    rank1_acc_overall: float,
    save_dir: Optional[str] = None
    ):
    """Plot evaluation results for the retrieval task for a specific subset."""

    scores = similarity_matrix.cpu()
    num_queries = scores.shape[0]

    # Calculate accuracy and collect scores *within the specified subset*
    subset_correct_scores = []
    subset_incorrect_scores = []
    subset_all_top_scores = [] # Scores of the top match (excluding self) for this subset

    num_subset_queries = 0
    num_subset_correct = 0

    print(f"\n--- Analyzing Subset: {name} ({len(input_indices)} potential queries) ---")

    for i in range(num_queries):
        if i not in input_indices:
            continue # Skip if query index not in the current subset

        num_subset_queries += 1
        query_inst_id = query_ids[i]

        # Find top match excluding self for Rank-1 eval
        similarities_rank1 = scores[i].clone()
        similarities_rank1[i] = -float('inf')
        best_match_idx_rank1 = torch.argmax(similarities_rank1).item()
        top_score_rank1 = similarities_rank1[best_match_idx_rank1].item()

        subset_all_top_scores.append(top_score_rank1)

        # Check if the top match (excluding self) belongs to the same instance ID
        if query_inst_id == gallery_ids[best_match_idx_rank1]:
            subset_correct_scores.append(top_score_rank1)
            num_subset_correct += 1
        else:
            subset_incorrect_scores.append(top_score_rank1)

    if num_subset_queries == 0:
        print(f"No samples found for subset '{name}'. Skipping plot.")
        return

    subset_acc = (num_subset_correct / num_subset_queries) * 100.0
    print(f"Subset '{name}': Rank-1 Accuracy = {subset_acc:.2f}% ({num_subset_correct}/{num_subset_queries})")

    # --- Plotting ---
    fig = plt.figure(figsize=(20, 6))
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    plt.suptitle(f"Retrieval Performance: {name.upper()} (Overall Acc: {rank1_acc_overall:.2f}%)", fontsize=16)

    # Plot scores of correctly identified top matches
    ax1 = plt.subplot(1, 3, 1, facecolor='#f0f0f0')
    if subset_correct_scores:
        plt.hist(subset_correct_scores, bins=50, range=(0.0, 1.0), label="Correct Top Matches", color='mediumseagreen', alpha=0.8)
    plt.xlabel('Similarity Score (Top Match, Excl. Self)')
    plt.ylabel('Frequency')
    plt.title(f"Correct Matches (N={len(subset_correct_scores)})", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    # Plot scores of incorrectly identified top matches
    ax2 = plt.subplot(1, 3, 2, facecolor='#f0f0f0')
    if subset_incorrect_scores:
        plt.hist(subset_incorrect_scores, bins=50, range=(0.0, 1.0), label="Incorrect Top Matches", color='lightcoral', alpha=0.8)
    plt.xlabel('Similarity Score (Top Match, Excl. Self)')
    plt.ylabel('Frequency')
    plt.title(f"Incorrect Matches (N={len(subset_incorrect_scores)})", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    # Plot all top-match scores for this subset
    ax3 = plt.subplot(1, 3, 3, facecolor='#f0f0f0')
    if subset_all_top_scores:
        plt.hist(subset_all_top_scores, bins=50, range=(0.0, 1.0), label="All Top Matches", color='cornflowerblue', alpha=0.8)
    plt.xlabel('Similarity Score (Top Match, Excl. Self)')
    plt.ylabel('Frequency')
    plt.title(f"All Top Matches ({name}) - Acc: {subset_acc:.2f}%", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout

    if save_dir:
        if not exists(save_dir):
            os.makedirs(save_dir)
        plot_filename = join(save_dir, f"retrieval_plot_{name.lower().replace(' ', '_')}.png")
        try:
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig) # Close the figure after saving to free memory
    else:
        plt.show() 