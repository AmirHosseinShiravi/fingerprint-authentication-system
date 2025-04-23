import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project modules
from . import config

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=config.EMBEDDING_DIM, input_size=config.RESNET_SIZE):
        """
        Initializes a custom Embedding Network.

        Args:
            embedding_dim (int): The desired dimension for the output embeddings.
            input_size (tuple): The expected input image size (Height, Width).
                                  Used to calculate the flattened feature size dynamically.
        """
        super().__init__()
        print(f"Initializing EmbeddingNet: embedding_dim={embedding_dim}, input_size={input_size}")

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Define the convolutional part of the network
        # Using padding=0 which is equivalent to padding='valid' in TF/Keras sense
        self.cnn_block = nn.Sequential(
            # Input: (B, 1, H, W) - Assuming 1 input channel (grayscale)
            nn.Conv2d(1, 16, kernel_size=3, padding=0), # Output: (B, 16, H-2, W-2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0), # Output: (B, 16, floor((H-2-2)/3), floor((W-2-2)/3)) -> check math

            nn.Conv2d(16, 32, kernel_size=3, padding=0), # Output: (B, 32, H_cnn1-2, W_cnn1-2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0), # Output: (B, 32, floor((H_cnn2-2-2)/3), floor((W_cnn2-2-2)/3))
        )

        # Calculate the output size of the CNN block dynamically
        self._num_features = self._get_cnn_output_size()
        print(f"Calculated CNN output features (flattened): {self._num_features}")

        if self._num_features <= 0:
             raise ValueError(f"CNN output feature size is {self._num_features}. "
                              f"Check input size {self.input_size} and CNN architecture (kernel sizes, padding, pooling). Maybe input too small?")

        # Define the fully connected part
        fc_layer1_out = embedding_dim * 2
        fc_layer2_out = embedding_dim * 3

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._num_features, fc_layer1_out),
            nn.ReLU(inplace=True),
            # Optional: Add Dropout or BatchNorm here if needed
            # nn.BatchNorm1d(fc_layer1_out),
            # nn.Dropout(0.3),
            nn.Linear(fc_layer1_out, fc_layer2_out),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(fc_layer2_out),
            # nn.Dropout(0.3),
            nn.Linear(fc_layer2_out, embedding_dim)
            # No final activation here, normalization happens in forward pass
        )

        # Optional: Initialize linear layers (e.g., Kaiming uniform)
        self._initialize_weights()

        # Verify total parameters
        # total_params = sum(p.numel() for p in self.parameters())
        # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total parameters in EmbeddingNet: {total_params:,}")
        # print(f"Trainable parameters: {trainable_params:,}")

    def _get_cnn_output_size(self):
        """Helper function to calculate the output size of the CNN block."""
        # Create a dummy input tensor with the expected size
        dummy_input = torch.zeros(1, 1, self.input_size[0], self.input_size[1])
        # Pass it through the CNN block
        with torch.no_grad(): # No need to track gradients
            dummy_output = self.cnn_block(dummy_input)
        # Calculate the total number of features after flattening
        num_features = dummy_output.numel() # numel() gives total elements (B*C*H*W)
        # Since B=1, this is C*H*W
        return num_features

    def _initialize_weights(self):
        """Initializes weights for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): # Initialize BatchNorm if used
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Expected Channels=1 for this network.

        Returns:
            torch.Tensor: L2-normalized embedding tensor of shape (Batch, embedding_dim).
        """
        # --- Input Check ---
        if x.shape[1] != 1:
             # Option 1: Raise error if input channels mismatch
             # raise ValueError(f"Expected 1 input channel, but got {x.shape[1]}")
             # Option 2: Try to average channels (use with caution)
             print(f"Warning: EmbeddingNet expects 1 input channel, got {x.shape[1]}. Averaging channels.")
             x = torch.mean(x, dim=1, keepdim=True)

        # Pass through CNN layers
        x = self.cnn_block(x)
        # Flatten and pass through FC layers
        embedding = self.fc_block(x)
        # L2 Normalize the final embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding 