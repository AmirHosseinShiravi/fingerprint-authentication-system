import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Any
import random
import time

# Import project modules
import utils
import config

# --- Triplet Fingerprint Dataset ---
class TripletFingerprintDataset(Dataset):
    def __init__(self,
                 unique_samples_dict: Dict[str, List[str]],
                 instance_ids: List[str],
                 base_transform: transforms.Compose,
                 augment_transform: Optional[transforms.Compose] = None,
                 load_all_images: bool = config.PRELOAD_IMAGES):
        """
        Initializes the Triplet Fingerprint Dataset.

        Args:
            unique_samples_dict: Dictionary mapping base_id to list of file paths.
            instance_ids: List of base_ids to include in this dataset split.
            base_transform: Basic transformations (ToTensor, Normalize) applied to all images.
            augment_transform: Augmentations applied only during training.
            load_all_images: Whether to preload all images into RAM.
        """
        # Filter the main dictionary to include only the instance IDs for this specific split
        self.unique_samples_dict = {k: v for k, v in unique_samples_dict.items() if k in instance_ids}
        self.instance_ids = instance_ids # These are the base_ids (unique fingers) for this split
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.is_train = augment_transform is not None
        self.images_cache = {}
        self.load_all_images = load_all_images

        # Pre-load images if requested (requires significant RAM)
        if self.load_all_images:
            print(f"Pre-loading images for {len(self.instance_ids)} instances into memory...")
            start_time = time.time()
            loaded_count = 0
            for base_id in self.instance_ids:
                self.images_cache[base_id] = {}
                for fpath in self.unique_samples_dict.get(base_id, []): # Use .get for safety
                     img = utils.load_image(fpath) # Use load_image from utils
                     if img:
                         self.images_cache[base_id][fpath] = img # Store PIL images
                         loaded_count += 1
            load_time = time.time() - start_time
            print(f"Loaded {loaded_count} images in {load_time:.2f} seconds.")

    def _get_image(self, file_path: str, base_id: Optional[str] = None) -> Optional[torch.Tensor]:
        """Loads or retrieves image and applies transformations."""
        img = None
        try:
            if self.load_all_images and base_id:
                img = self.images_cache.get(base_id, {}).get(file_path)
                if img is None: # If somehow not preloaded, load now
                     img = utils.load_image(file_path)
                     # Optionally cache it now if missing during preload
                     # if img and base_id:
                     #    if base_id not in self.images_cache: self.images_cache[base_id] = {}
                     #    self.images_cache[base_id][file_path] = img
            else:
                img = utils.load_image(file_path) # Load PIL image on the fly

            if img is None:
                # print(f"Warning: Could not load image {file_path}. Skipping.")
                return None

            # Apply augmentations FIRST if it's training phase (on PIL image)
            if self.is_train and self.augment_transform:
                 img = self.augment_transform(img)

            # Apply base transforms (ToTensor, Normalize)
            img_tensor = self.base_transform(img)
            return img_tensor

        except Exception as e:
             print(f"Error processing image {file_path} in _get_image: {e}")
             return None

    def __len__(self) -> int:
        # The length is the number of unique finger instances (potential anchors)
        return len(self.instance_ids)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        # Retry mechanism for recoverable errors (like file not found or loading issue)
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Adjust index if retrying to avoid getting stuck on the same problematic item
                current_idx = (idx + retry) % len(self)
                anchor_instance_id = self.instance_ids[current_idx]

                possible_paths = self.unique_samples_dict.get(anchor_instance_id)

                # Check if instance exists and has enough variations (at least 2)
                if not possible_paths or len(possible_paths) < 2:
                    if retry == max_retries - 1: # Log error only on last retry
                        print(f"Warning: Instance {anchor_instance_id} has < 2 variations after retries. Skipping index {idx}.")
                    continue # Try next index/retry

                # Select Anchor and Positive from the *same* finger instance
                anchor_path, positive_path = random.sample(possible_paths, 2)

                # Select Negative Instance (Different Finger Instance from the current split)
                negative_instance_id = anchor_instance_id
                loop_count = 0
                while negative_instance_id == anchor_instance_id:
                    negative_instance_id = random.choice(self.instance_ids)
                    loop_count += 1
                    if loop_count > len(self.instance_ids) * 2: # Safety break
                        print(f"Warning: Could not find a different negative ID for {anchor_instance_id}. Skipping index {idx}. Check dataset split.")
                        return None # Cannot form a valid triplet

                negative_possible_paths = self.unique_samples_dict.get(negative_instance_id)
                if not negative_possible_paths: # Check if negative ID somehow has no paths
                     if retry == max_retries - 1:
                          print(f"Warning: Negative instance {negative_instance_id} has no paths. Skipping index {idx}. Check dataset integrity.")
                     continue # Try next index/retry

                negative_path = random.choice(negative_possible_paths)

                # Load and Transform Images
                anchor_img = self._get_image(anchor_path, anchor_instance_id)
                positive_img = self._get_image(positive_path, anchor_instance_id)
                negative_img = self._get_image(negative_path, negative_instance_id)

                # Handle potential loading errors from _get_image
                if anchor_img is None or positive_img is None or negative_img is None:
                    if retry == max_retries - 1:
                         print(f"Warning: Failed to load one or more images for triplet anchor {anchor_instance_id} after retries. Skipping index {idx}. Check image paths and loading function.")
                    continue # Try next index/retry

                # Success! Return the triplet and the anchor's instance ID
                return anchor_img, positive_img, negative_img, anchor_instance_id

            except Exception as e: # Catch broader errors during item fetching
                 print(f"Error in TripletFingerprintDataset __getitem__ for index {idx}, retry {retry+1}/{max_retries}: {e}")
                 if retry == max_retries - 1:
                      print(f"Failed to get item for index {idx} after {max_retries} retries.")
                      return None # Give up after max retries

        # If all retries failed
        return None


# --- Collate Function for Triplet Loader ---
def collate_fn(batch: List[Optional[Tuple[Any, ...]]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]:
    """Filters out None items from a batch and stacks tensors."""
    # Filter out None values first (handles errors in __getitem__)
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return None if the batch is empty after filtering
        return None

    try:
        # Unzip the batch (list of tuples) into separate lists
        anchors, positives, negatives, anchor_ids = zip(*batch)

        # Stack tensors - this assumes all tensors in the batch have the same shape
        anchor_batch = torch.stack(anchors)
        positive_batch = torch.stack(positives)
        negative_batch = torch.stack(negatives)

        # anchor_ids remain as a list of strings
        return anchor_batch, positive_batch, negative_batch, list(anchor_ids)
    except Exception as e:
        print(f"Error during collation (collate_fn): {e}. Batch items (first 5): {batch[:5]}")
        # Return None if stacking fails
        return None

# --- Dataset for Embedding Extraction ---
class EmbeddingExtractionDataset(Dataset):
    def __init__(self, unique_samples_dict, instance_ids, base_transform):
        """
        Dataset specifically for extracting embeddings from all variations in a split.
        """
        self.image_paths = []
        self.instance_ids_map = [] # Map index to instance_id
        self.base_transform = base_transform
        instance_ids_set = set(instance_ids) # Use set for faster lookups

        print("Building embedding extraction dataset...")
        count = 0
        for inst_id, paths in unique_samples_dict.items():
             if inst_id in instance_ids_set:
                for path in paths:
                     self.image_paths.append(path)
                     self.instance_ids_map.append(inst_id)
                     count+=1
        print(f"Found {count} unique image variations in the target split.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
         path = self.image_paths[idx]
         instance_id = self.instance_ids_map[idx]
         try:
             img = utils.load_image(path)
             if img is None: return None, None, None # Handle loading failure
             img_tensor = self.base_transform(img)
             return img_tensor, instance_id, path # Return tensor, id, and path
         except Exception as e:
              print(f"Error loading image {path} for embedding extraction: {e}")
              return None, None, None

# --- Collate Function for Embedding Extraction Loader ---
def embed_collate_fn(batch):
    """Collates data for the EmbeddingExtractionDataset.

    Filters None items and returns stacked tensors, list of IDs, and list of paths.
    """
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None # Return None if batch is empty
    try:
        tensors, ids, paths = zip(*batch)
        return torch.stack(tensors), list(ids), list(paths)
    except Exception as e:
        print(f"Error during collation (embed_collate_fn): {e}")
        return None # Return None on error 