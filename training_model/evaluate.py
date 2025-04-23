import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict
import time
import random
from os.path import join, exists, dirname
import os
import re
from collections import defaultdict

# Import project modules
import config
import utils
import dataset # Import dataset module for EmbeddingExtractionDataset and collate_fn
from model import EmbeddingNet # Import the model

@torch.no_grad()
def evaluate_model_retrieval(
    model: nn.Module,
    test_loader: DataLoader, # Should use EmbeddingExtractionDataset based loader
    device: torch.device
    ) -> Optional[Tuple[torch.Tensor, List[str], List[str], float]]:
    """
    Evaluate the model using a retrieval task on the test set.
    Extracts embeddings for all samples in the test_loader, calculates
    cosine similarity, and computes Rank-1 identification accuracy.

    Args:
        model: The trained embedding model.
        test_loader: DataLoader containing the test set (using EmbeddingExtractionDataset).
        device: The device to run evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        A tuple containing:
            - similarity_matrix (torch.Tensor): Pairwise cosine similarities.
            - query_ids (List[str]): Instance IDs corresponding to rows/queries.
            - query_paths (List[str]): File paths corresponding to rows/queries.
            - rank1_accuracy (float): The calculated Rank-1 accuracy.
        Returns None if evaluation cannot be performed (e.g., no embeddings extracted).
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_instance_ids = []
    all_paths = [] # Store the path for each embedding

    print("\n--- Extracting Test Set Embeddings ---")
    processed_count = 0
    extraction_start_time = time.time()

    # Iterate through the test_loader (which should use embed_collate_fn)
    for batch_data in test_loader:
        if batch_data is None or batch_data[0] is None:
            # print("Warning: Skipping empty/invalid batch in embedding extraction.")
            continue

        images, batch_instance_ids, batch_paths = batch_data

        if images is None or not batch_instance_ids:
             print("Warning: Skipping batch with None images or empty IDs.")
             continue

        images = images.to(device, non_blocking=True)

        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_instance_ids.extend(batch_instance_ids)
        all_paths.extend(batch_paths)

        processed_count += images.size(0)
        if processed_count % (config.EVAL_BATCH_SIZE * 10) == 0 and processed_count > 0:
            print(f"  Extracted embeddings for {processed_count}/{len(test_loader.dataset)} images...")

    if not all_embeddings:
        print("Error: No embeddings extracted from the test set. Cannot evaluate.")
        return None

    all_embeddings = torch.cat(all_embeddings, dim=0)
    extraction_duration = time.time() - extraction_start_time
    print(f"--- Embedding Extraction Complete ({all_embeddings.shape[0]} embeddings) in {extraction_duration:.2f}s ---")

    # --- Perform Retrieval (Rank-1 Accuracy) ---
    print("--- Performing Retrieval Evaluation ---")
    retrieval_start_time = time.time()

    # Use all extracted embeddings as both query and gallery
    gallery_embeddings = all_embeddings
    gallery_ids = all_instance_ids

    query_embeddings = all_embeddings
    query_ids = all_instance_ids
    query_paths = all_paths # Paths associated with queries

    print(f"Calculating similarity matrix ({query_embeddings.shape[0]} x {gallery_embeddings.shape[0]})...")
    # Calculate cosine similarity (embeddings are L2 normalized)
    # Move calculation to GPU if possible for speed
    similarity_matrix = torch.matmul(query_embeddings.to(device), gallery_embeddings.t().to(device)).cpu()
    print("Similarity matrix calculation complete.")

    num_queries = similarity_matrix.shape[0]
    correct_rank1 = 0
    # matched_indices = [] # Store the index of the top match for each query (excluding self)

    print("Finding top matches and calculating Rank-1 accuracy...")
    for i in range(num_queries):
        similarities_rank1 = similarity_matrix[i].clone()
        similarities_rank1[i] = -float('inf') # Exclude exact self-match

        # Check if all other similarities are -inf (can happen in very small datasets)
        if torch.isinf(similarities_rank1).all():
            # print(f"Warning: Query {i} (ID: {query_ids[i]}) has no valid non-self matches. Skipping Rank-1 check for this query.")
            best_match_idx_rank1 = i # Assign self index, won't be counted as correct
        else:
            best_match_idx_rank1 = torch.argmax(similarities_rank1).item()

        # matched_indices.append(best_match_idx_rank1)

        # Is the instance ID of the top match (excluding self) the same as the query instance ID?
        if query_ids[i] == gallery_ids[best_match_idx_rank1]:
             correct_rank1 += 1

        if i > 0 and i % 1000 == 0:
             print(f"  Processed {i}/{num_queries} queries...")

    rank1_accuracy = (correct_rank1 / num_queries) * 100.0 if num_queries > 0 else 0.0
    retrieval_duration = time.time() - retrieval_start_time
    print(f"\n--- Evaluation Complete (in {retrieval_duration:.2f}s) ---")
    print(f"Rank-1 Identification Accuracy: {rank1_accuracy:.2f}% ({correct_rank1}/{num_queries})")

    # Return information needed for detailed analysis and plotting
    return similarity_matrix, query_ids, query_paths, rank1_accuracy


def get_difficulty_indices(query_paths: List[str]) -> Dict[str, set]:
    """Categorizes query indices based on filename patterns (Real, Easy, Medium, Hard)."""
    indices = {
        "Real": set(),
        "Altered-Easy": set(),
        "Altered-Medium": set(),
        "Altered-Hard": set()
    }
    # Regex to find difficulty level in the path (adjust if path structure differs)
    # Assumes paths like "/.../SOCOFing/Real/file.bmp" or "/.../SOCOFing/Altered/Altered-Easy/file.bmp"
    real_pattern = re.compile(r"/Real/")
    easy_pattern = re.compile(r"/Altered-Easy/")
    medium_pattern = re.compile(r"/Altered-Medium/")
    hard_pattern = re.compile(r"/Altered-Hard/")

    for i, path in enumerate(query_paths):
        path_norm = path.replace("\\", "/") # Normalize path separators
        if real_pattern.search(path_norm):
            indices["Real"].add(i)
        elif easy_pattern.search(path_norm):
            indices["Altered-Easy"].add(i)
        elif medium_pattern.search(path_norm):
            indices["Altered-Medium"].add(i)
        elif hard_pattern.search(path_norm):
            indices["Altered-Hard"].add(i)
        # else: print(f"Warning: Could not categorize path: {path}")

    return indices

if __name__ == '__main__':
    print(f"Using device: {config.DEVICE}")

    # --- 1. Load Data (Only Test Set Needed) ---
    print("\n--- Preparing Test Data ---")
    prep_start_time = time.time()

    # Need to reconstruct the unique_sample dictionary to build the test dataset
    # Assume dataset structure as defined in config and utils
    real_path = join(config.SRC_FOLDER, "Real")
    altered_easy_path = join(config.SRC_FOLDER, "Altered", "Altered-Easy")
    altered_medium_path = join(config.SRC_FOLDER, "Altered", "Altered-Medium")
    altered_hard_path = join(config.SRC_FOLDER, "Altered", "Altered-Hard")

    # Check paths exist
    for path in [config.SRC_FOLDER, real_path, altered_easy_path, altered_medium_path, altered_hard_path]:
        if not exists(path):
            raise FileNotFoundError(
                f"Required dataset folder not found: {path}. "
                f"Please check the SRC_FOLDER path in config.py and ensure the dataset is structured correctly."
            )

    real_filenames = utils.list_files(real_path)
    easy_filenames = utils.list_files(altered_easy_path)
    medium_filenames = utils.list_files(altered_medium_path)
    hard_filenames = utils.list_files(altered_hard_path)

    if not real_filenames and not easy_filenames and not medium_filenames and not hard_filenames:
         raise ValueError(f"No image files found in the dataset subdirectories under {config.SRC_FOLDER}. Check dataset integrity.")

    unique_sample = defaultdict(list)
    file_count = 0
    for fn in real_filenames:
        base_id = utils.get_base_id(fn, is_real=True)
        unique_sample[base_id].append(join(real_path, fn))
        file_count+=1
    for fn in easy_filenames:
        base_id = utils.get_base_id(fn, is_real=False)
        unique_sample[base_id].append(join(altered_easy_path, fn))
        file_count+=1
    for fn in medium_filenames:
        base_id = utils.get_base_id(fn, is_real=False)
        unique_sample[base_id].append(join(altered_medium_path, fn))
        file_count+=1
    for fn in hard_filenames:
        base_id = utils.get_base_id(fn, is_real=False)
        unique_sample[base_id].append(join(altered_hard_path, fn))
        file_count+=1

    print(f"Found {file_count} total files.")
    print(f"Grouped into {len(unique_sample)} unique finger instances.")

    # Filter out fingers with only one variation (cannot be used in triplet training, but maybe ok for eval?)
    # Keep them for evaluation to have a complete gallery
    unique_sample_filtered = {k: v for k, v in unique_sample.items() if len(v) >= 1} # Require at least 1 sample for gallery/query
    print(f"Using {len(unique_sample_filtered)} instances for evaluation dataset.")

    if not unique_sample_filtered:
        raise ValueError("No valid finger instances found for evaluation.")

    # Split instance IDs - REPRODUCE the same split as in training!
    finger_instance_ids = sorted(list(unique_sample_filtered.keys()))
    random.seed(42) # Use the same seed as training
    random.shuffle(finger_instance_ids)

    total_instances = len(finger_instance_ids)
    train_split = int(0.7 * total_instances)
    val_split = int(0.85 * total_instances)

    # training_instance_ids = finger_instance_ids[:train_split]
    # validation_instance_ids = finger_instance_ids[train_split:val_split]
    test_instance_ids = finger_instance_ids[val_split:]

    # Adjust splits if empty (copy logic from training script for consistency)
    if not test_instance_ids and finger_instance_ids:
        print("Warning: Test split is empty based on standard ratios. Re-allocating from train/val splits might be needed if this is unexpected, or evaluation cannot proceed.")
        # Decide how to handle - maybe take last 15% regardless?
        test_split_idx = max(0, total_instances - int(0.15 * total_instances)) # Ensure at least 15%
        test_instance_ids = finger_instance_ids[test_split_idx:]
        if not test_instance_ids and total_instances > 0:
             test_instance_ids = [finger_instance_ids[-1]] # Take at least one if possible

    print(f"Using {len(test_instance_ids)} instances for the test set.")
    if not test_instance_ids:
         raise ValueError("No instances allocated to the test set after splitting. Cannot evaluate.")

    prep_duration = time.time() - prep_start_time
    print(f"--- Test Data Preparation Complete ({prep_duration:.2f}s) ---")

    # --- 2. Create Test Dataset and DataLoader for Embedding Extraction ---
    print("\n--- Creating Test DataLoader for Embedding Extraction ---")
    eval_dataset = dataset.EmbeddingExtractionDataset(
        unique_samples_dict=unique_sample_filtered,
        instance_ids=test_instance_ids,
        base_transform=utils.base_transform # Use the base transform, no augmentation
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.EVAL_BATCH_SIZE, # Use evaluation batch size
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dataset.embed_collate_fn # Use the collation function for embedding extraction
    )
    print(f"Test DataLoader created with {len(eval_loader)} batches.")

    # --- 3. Load Model ---
    print("\n--- Loading Model ---")
    model = EmbeddingNet(embedding_dim=config.EMBEDDING_DIM, input_size=config.RESNET_SIZE)
    # Find the best checkpoint file
    checkpoint_path = join(config.CHECKPOINT_DIR, 'best_model.pth')
    if not exists(checkpoint_path):
        # Try finding the latest epoch checkpoint if best doesn't exist
        epoch_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('epoch_') and f.endswith('_model.pth')]
        if epoch_files:
            # Define a function to extract epoch number safely
            def get_epoch_num(filename):
                match = re.search(r'epoch_(\d+)_model', filename)
                return int(match.group(1)) if match else -1 # Return -1 or some indicator if pattern not found

            # Sort using the safe function, filter out any files that didn't match
            valid_epoch_files = [f for f in epoch_files if get_epoch_num(f) != -1]
            if not valid_epoch_files:
                 raise FileNotFoundError(f"No valid epoch checkpoint files found in {config.CHECKPOINT_DIR}. Pattern 'epoch_(\\d+)_model.pth' might not match.")

            latest_epoch_file = sorted(valid_epoch_files, key=get_epoch_num, reverse=True)[0]
            checkpoint_path = join(config.CHECKPOINT_DIR, latest_epoch_file)
            print(f"Warning: 'best_model.pth' not found. Loading latest epoch checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No model checkpoint found in {config.CHECKPOINT_DIR}. Please train the model first.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        # Check for embedding dimension consistency
        if 'embedding_dim' in checkpoint and checkpoint['embedding_dim'] != config.EMBEDDING_DIM:
            print(f"Warning: Checkpoint embedding dimension ({checkpoint['embedding_dim']}) differs from config ({config.EMBEDDING_DIM}).")

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
        if 'epoch' in checkpoint:
             print(f"Checkpoint trained for {checkpoint['epoch']} epochs.")
        if 'val_loss' in checkpoint:
             print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")

    except KeyError:
        # Handle older checkpoints that might not have all keys
        print("Warning: Checkpoint missing some keys (e.g., 'epoch', 'val_loss', 'embedding_dim'). Loading model state dict only.")
        try:
            # Attempt to load just the state dict if the structure is different
            model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
            print(f"Loaded model weights from {checkpoint_path} (basic load).")
        except Exception as load_err:
            raise RuntimeError(f"Failed to load model state_dict from {checkpoint_path}: {load_err}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")

    # --- 4. Evaluate Model ---
    evaluation_results = evaluate_model_retrieval(
        model=model,
        test_loader=eval_loader,
        device=config.DEVICE
    )

    # --- 5. Analyze and Plot Results ---
    if evaluation_results:
        similarity_matrix, query_ids, query_paths, rank1_acc = evaluation_results
        print("\n--- Analyzing Results by Difficulty ---")

        # Determine gallery IDs from the query IDs/paths if necessary (assuming gallery == query set)
        gallery_ids = query_ids

        # Get indices based on query path difficulty
        difficulty_indices = get_difficulty_indices(query_paths)

        # Define plot save directory
        plot_save_dir = join(dirname(__file__), "evaluation_plots") # Save plots in a subdir

        # Plot results for each difficulty subset
        for difficulty_name, indices_set in difficulty_indices.items():
            if indices_set: # Only plot if there are samples in the subset
                utils.plot_retrieval_results(
                    name=difficulty_name,
                    input_indices=indices_set,
                    similarity_matrix=similarity_matrix,
                    query_ids=query_ids,
                    query_paths=query_paths,
                    gallery_ids=gallery_ids,
                    rank1_acc_overall=rank1_acc,
                    save_dir=plot_save_dir # Pass save directory
                )
            else:
                 print(f"Skipping plot for '{difficulty_name}' as no samples were found.")

        # Optionally plot for the overall test set
        print("\n--- Plotting Overall Test Set Results ---")
        utils.plot_retrieval_results(
            name="Overall Test Set",
            input_indices=set(range(len(query_ids))), # All indices
            similarity_matrix=similarity_matrix,
            query_ids=query_ids,
            query_paths=query_paths,
            gallery_ids=gallery_ids,
            rank1_acc_overall=rank1_acc,
            save_dir=plot_save_dir
        )
    else:
        print("Evaluation failed, skipping result analysis and plotting.") 