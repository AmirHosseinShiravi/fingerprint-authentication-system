import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import cv2

# Add the parent directory to sys.path to import from training_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our training model code
from training_model.model import EmbeddingNet
from training_model.utils import load_image, normalize, base_transform

class FingerprintModel:
    def __init__(self, model_path=None, embedding_dim=128, input_size=(96, 96)):
        """
        Initialize the FingerprintModel with our custom EmbeddingNet.
        
        Args:
            model_path: Path to the trained model checkpoint
            embedding_dim: Dimension of the embedding vector
            input_size: Input image size expected by the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        
        # Initialize the model
        self.model = EmbeddingNet(embedding_dim=embedding_dim, input_size=input_size).to(self.device)
        self.model.eval()
        
        # Load trained model weights if provided
        if model_path:
            self._load_model(model_path)
        
        # Set up transformations (using our utility functions)
        self.transform = base_transform
        
    def _load_model(self, model_path):
        """Load the model weights from a checkpoint file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess an image for model input.
        
        Args:
            image_path_or_array: Path to image file or numpy array containing image data
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert input to PIL Image
        if isinstance(image_path_or_array, str):
            # Use our utility function to load image properly
            img = load_image(image_path_or_array)
            if img is None:
                raise ValueError(f"Failed to load image from {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image_path_or_array.shape) == 2:  # Grayscale
                img = Image.fromarray(image_path_or_array)
            elif len(image_path_or_array.shape) == 3:  # RGB or RGBA
                img = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError(f"Unsupported image shape: {image_path_or_array.shape}")
                
            # Resize to expected size
            img = img.convert('L')  # Convert to grayscale
            img = img.resize(self.input_size)
        else:
            raise TypeError(f"Unsupported image type: {type(image_path_or_array)}")
        
        # Apply transformation - this should convert the PIL Image to a tensor
        tensor = self.transform(img)
        
        # Ensure we have a tensor and add batch dimension
        if not isinstance(tensor, torch.Tensor):
            # If for some reason we don't have a tensor, convert manually
            tensor = transforms.ToTensor()(img)
            tensor = normalize(tensor)
            
        # Add batch dimension    
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def extract_features(self, image_path_or_array):
        """
        Extract fingerprint embedding features from an image.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        with torch.no_grad():
            image = self.preprocess_image(image_path_or_array)
            features = self.model(image)
            return features.cpu().numpy()
    
    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            features1, features2: Feature vectors
            
        Returns:
            float: Similarity score (0-1)
        """
        # Ensure features are normalized
        if isinstance(features1, np.ndarray) and isinstance(features2, np.ndarray):
            # Compute dot product (cosine similarity for L2 normalized vectors)
            similarity = np.dot(features1, features2.T)
            return similarity[0][0] if similarity.shape == (1, 1) else similarity
        else:
            raise TypeError("Features must be numpy arrays")
    
    def verify_fingerprint(self, image1_path_or_array, image2_path_or_array, threshold=0.75):
        """
        Verify if two fingerprint images match.
        
        Args:
            image1_path_or_array, image2_path_or_array: Paths or arrays of images to compare
            threshold: Similarity threshold for match decision
            
        Returns:
            tuple: (is_match, similarity_score)
        """
        features1 = self.extract_features(image1_path_or_array)
        features2 = self.extract_features(image2_path_or_array)
        similarity = self.compute_similarity(features1, features2)
        return similarity > threshold, similarity 