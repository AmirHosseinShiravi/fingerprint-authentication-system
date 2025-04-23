import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import time
import os
from os.path import join, exists, dirname
from collections import defaultdict
import random
import torch.nn.functional as F

# Import project modules
import config
import utils
import dataset
from model import EmbeddingNet

def train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    log_interval: int
    ) -> float:
    """Runs a single training epoch."""
    model.train()
    train_loss = 0.0
    batches_processed = 0
    epoch_start_time = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        # Handle potential None batches from collate_fn
        if batch_data is None or batch_data[0] is None:
            # print(f"Warning: Skipping empty/invalid batch {batch_idx} in training epoch {epoch_num}.")
            continue

        anchors, positives, negatives, _ = batch_data # Ignore anchor_ids in training loop

        # Move data to the appropriate device
        # Use non_blocking=True for potentially faster transfer if using CUDA & pinned memory
        anchors = anchors.to(device, non_blocking=True)
        positives = positives.to(device, non_blocking=True)
        negatives = negatives.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass: Get embeddings
        emb_a = model(anchors)
        emb_p = model(positives)
        emb_n = model(negatives)

        # Compute triplet loss
        loss = loss_fn(emb_a, emb_p, emb_n)

        # Backward pass
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        batches_processed += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch_num}/{total_epochs}, Batch: {batch_idx}/{len(train_loader)}, ' \
                  f'Avg Batch Loss: {train_loss/batches_processed:.4f}, LR: {current_lr:.6f}')

    # Guard against empty loader for an epoch
    if batches_processed == 0:
         print(f"Epoch {epoch_num}: No batches processed in training. Check DataLoader/Dataset.")
         return float('nan') # Return NaN or raise error

    avg_train_loss = train_loss / batches_processed
    epoch_duration = time.time() - epoch_start_time
    print(f'--- Epoch {epoch_num} Training Complete --- Avg Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_train_loss

def validate_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    margin: float
    ) -> Tuple[float, float, float, float]:
    """Runs a single validation epoch."""
    model.eval()
    val_loss = 0.0
    val_batches_processed = 0
    positive_dist_total = 0.0
    negative_dist_total = 0.0
    num_active_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data is None or batch_data[0] is None:
                # print(f"Warning: Skipping empty/invalid batch in validation.")
                continue

            anchors, positives, negatives, _ = batch_data
            anchors = anchors.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)

            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = loss_fn(emb_a, emb_p, emb_n)
            val_loss += loss.item()
            val_batches_processed += 1

            # Calculate distances and fraction of active triplets
            dist_pos = F.pairwise_distance(emb_a, emb_p, p=2)
            dist_neg = F.pairwise_distance(emb_a, emb_n, p=2)
            positive_dist_total += dist_pos.sum().item()
            negative_dist_total += dist_neg.sum().item()

            # Triplets violating the margin (d(a,p) + margin > d(a,n))
            active = (dist_pos + margin > dist_neg)
            num_active_triplets += active.sum().item()
            total_triplets += anchors.size(0)

    if val_batches_processed == 0:
        print("Warning: No batches processed in validation. Cannot evaluate model performance.")
        return float('inf'), 0.0, 0.0, 0.0 # Return high loss and zero stats

    avg_val_loss = val_loss / val_batches_processed
    avg_pos_dist = positive_dist_total / total_triplets if total_triplets > 0 else 0
    avg_neg_dist = negative_dist_total / total_triplets if total_triplets > 0 else 0
    fraction_active = num_active_triplets / total_triplets if total_triplets > 0 else 0

    print(f'--- Validation Complete --- Avg Loss: {avg_val_loss:.4f}')
    print(f'  Avg Pos Distance: {avg_pos_dist:.4f}')
    print(f'  Avg Neg Distance: {avg_neg_dist:.4f}')
    print(f'  Fraction Active Triplets: {fraction_active:.4f}')

    return avg_val_loss, avg_pos_dist, avg_neg_dist, fraction_active

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: float,
    is_best: bool,
    output_dir: str
    ):
    """Saves model checkpoint."""
    if not exists(output_dir):
        os.makedirs(output_dir)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        'embedding_dim': config.EMBEDDING_DIM
    }
    filename = join(output_dir, f'epoch_{epoch}_model.pth')
    torch.save(state, filename)
    print(f"Saved epoch checkpoint to {filename}")

    if is_best:
        best_filename = join(output_dir, 'best_model.pth')
        torch.save(state, best_filename)
        print(f"*** Saved best model checkpoint to {best_filename} (Val Loss: {loss:.4f}) ***")


if __name__ == '__main__':
    print(f"Using device: {config.DEVICE}")
    print(f"Image resize dimensions: {config.RESNET_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Triplet margin: {config.MARGIN}")
    print(f"Preload images: {config.PRELOAD_IMAGES}")
    print(f"Checkpoint directory: {config.CHECKPOINT_DIR}")
    print(f"Num workers: {config.NUM_WORKERS}")

    # --- 1. Data Preparation ---
    print("\n--- Starting Data Preparation ---")
    prep_start_time = time.time()

    # Construct paths based on config
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

    # Collect filenames using utils
    real_filenames = utils.list_files(real_path)
    easy_filenames = utils.list_files(altered_easy_path)
    medium_filenames = utils.list_files(altered_medium_path)
    hard_filenames = utils.list_files(altered_hard_path)

    if not real_filenames and not easy_filenames and not medium_filenames and not hard_filenames:
         raise ValueError(f"No image files found in the dataset subdirectories under {config.SRC_FOLDER}. Check dataset integrity.")

    # Group fingerprint variations by base ID
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

    # Filter out fingers with only one variation (cannot form positive pairs for triplet loss)
    original_count = len(unique_sample)
    unique_sample_filtered = {k: v for k, v in unique_sample.items() if len(v) >= 2}
    filtered_count = original_count - len(unique_sample_filtered)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} instances with < 2 variations.")
    print(f"Using {len(unique_sample_filtered)} unique finger instances for dataset creation.")

    if not unique_sample_filtered:
        raise ValueError("No unique finger instances with >= 2 variations found. Cannot create triplets.")

    # Split instance IDs into train, validation, test sets
    finger_instance_ids = sorted(list(unique_sample_filtered.keys()))
    random.seed(42) # Ensure consistent splits
    random.shuffle(finger_instance_ids)

    total_instances = len(finger_instance_ids)
    train_split = int(0.7 * total_instances)
    val_split = int(0.85 * total_instances)

    training_instance_ids = finger_instance_ids[:train_split]
    validation_instance_ids = finger_instance_ids[train_split:val_split]
    test_instance_ids = finger_instance_ids[val_split:] # Keep track for reporting, though not used in training

    # Adjust if validation or test set is empty after split (can happen with small datasets)
    if not validation_instance_ids and test_instance_ids:
        print("Warning: Validation set empty after split. Moving half from test set.")
        move_count = max(1, len(test_instance_ids)//2)
        validation_instance_ids = test_instance_ids[:move_count]
        test_instance_ids = test_instance_ids[move_count:]
    elif not validation_instance_ids and training_instance_ids:
        print("Warning: Validation set empty, no test set to borrow from. Moving 10% from training.")
        move_count = max(1, int(0.1 * len(training_instance_ids)))
        validation_instance_ids = training_instance_ids[-move_count:]
        training_instance_ids = training_instance_ids[:-move_count]

    if not training_instance_ids:
         raise ValueError("Training set is empty after splitting. Check dataset size and split ratios.")

    print(f"Splitting instances into:")
    print(f"  Training: {len(training_instance_ids)}")
    print(f"  Validation: {len(validation_instance_ids)}")
    print(f"  Test (for reference): {len(test_instance_ids)}")
    prep_duration = time.time() - prep_start_time
    print(f"--- Data Preparation Complete ({prep_duration:.2f}s) ---")

    # --- 2. Create Datasets and DataLoaders ---
    print("\n--- Creating Datasets and DataLoaders ---")
    loader_start_time = time.time()

    train_dataset = dataset.TripletFingerprintDataset(
        unique_samples_dict=unique_sample_filtered,
        instance_ids=training_instance_ids,
        base_transform=utils.base_transform,
        augment_transform=utils.train_augment_transform,
        load_all_images=config.PRELOAD_IMAGES
    )
    val_dataset = dataset.TripletFingerprintDataset(
        unique_samples_dict=unique_sample_filtered,
        instance_ids=validation_instance_ids,
        base_transform=utils.base_transform,
        augment_transform=None, # No augmentation for validation
        load_all_images=config.PRELOAD_IMAGES
    )

    # Important: Set num_workers=0 for Windows compatibility if needed
    actual_num_workers = config.NUM_WORKERS
    if os.name == 'nt' and config.NUM_WORKERS > 0:
        print(f"Warning: Setting num_workers=0 for DataLoader on Windows (was {config.NUM_WORKERS}).")
        actual_num_workers = 0

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=actual_num_workers,
        pin_memory=True, # Set to True if using CUDA for faster host-to-device transfers
        collate_fn=dataset.collate_fn, # Use custom collate function
        drop_last=True # Drop last incomplete batch in training for potentially more stable training
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, # Larger batch size for validation
        num_workers=actual_num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    loader_duration = time.time() - loader_start_time
    print(f"DataLoaders created.")
    print(f"  Training batches per epoch: ~{len(train_loader)}")
    print(f"  Validation batches per epoch: ~{len(val_loader)}")
    print(f"--- DataLoader Creation Complete ({loader_duration:.2f}s) ---")

    # --- 3. Initialize Model, Loss, Optimizer, Scheduler ---
    print("\n--- Initializing Model, Loss, Optimizer ---")
    model = EmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        input_size=config.RESNET_SIZE
    )
    model.to(config.DEVICE)

    # Use standard TripletMarginLoss
    triplet_loss_fn = nn.TripletMarginLoss(margin=config.MARGIN, p=2, reduction='mean')

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Learning Rate Scheduler (ReduceLROnPlateau monitors validation loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # --- 4. Training Loop ---
    print(f"\n--- Starting Training on {config.DEVICE} for {config.NUM_EPOCHS} Epochs ---")
    best_val_loss = float('inf')
    training_start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print("-" * 60)
        train_loss = train_epoch(
            model, triplet_loss_fn, train_loader, optimizer, config.DEVICE, epoch, config.NUM_EPOCHS, config.LOG_INTERVAL
        )

        avg_val_loss, _, _, _ = validate_epoch(
            model, triplet_loss_fn, val_loader, config.DEVICE, config.MARGIN
        )

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss=avg_val_loss,
            is_best=is_best,
            output_dir=config.CHECKPOINT_DIR
        )

    training_duration = time.time() - training_start_time
    print(f"\n--- Training Finished --- Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Training Duration: {training_duration / 60:.2f} minutes") 