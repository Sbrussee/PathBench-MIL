#!/usr/bin/env python3
"""
slide_level_predictors.py

This module defines slide-level predictors for tasks such as classification,
survival analysis, or regression. These predictors operate on batched slide-level
embeddings (i.e. one aggregated feature vector per slide) and use helper functions
from aggregators.py to build encoders, select activation functions, and initialize weights.

Classes:
    linear_slide_classifier: A linear mapper for slide-level embeddings.
    mlp_slide_classifier: A multi-layer perceptron (MLP) for slide-level embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional
from pathbench.models.aggregators import get_activation_function, initialize_weights

def build_slide_level_encoder(n_feats: int,
                              z_dim: int,
                              encoder_layers: int,
                              activation_function: str,
                              dropout_p: float = 0.1,
                              use_batchnorm: bool = True):
    """
    Build a multi-layer encoder network.

    The encoder is composed of repeated blocks of:
      - Linear layer (input: either n_feats or z_dim, output: z_dim)
      - Optional Activation function (if activation_function is provided)
      - Optional Batch Normalization (if use_batchnorm is True; always using BatchNorm1d)
      - Optional Dropout (if dropout_p > 0)

    Args:
        n_feats (int): Number of input features.
        z_dim (int): Dimension of the latent representation.
        encoder_layers (int): Number of repeated blocks/layers.
        activation_function (str): Name of the activation function (e.g., 'ReLU').
        dropout_p (float): Dropout probability.
        use_batchnorm (bool): Whether to include BatchNorm layers.

    Returns:
        nn.Sequential: The constructed encoder network.
    """
    layers = []
    for i in range(encoder_layers):
        in_features = n_feats if i == 0 else z_dim
        layers.append(nn.Linear(in_features, z_dim))
        if activation_function is not None:
            layers.append(get_activation_function(activation_function))
        if use_batchnorm:
            # Always use BatchNorm1d to normalize over the feature dimension.
            layers.append(nn.BatchNorm1d(z_dim))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Slide-Level Predictor Models
# -----------------------------------------------------------------------------

class linear_slide_classifier(nn.Module):
    """
    linear_slide_classifier is a simple linear mapper for slide-level embeddings.
    
    This model applies a single linear transformation to the input embedding.
    The final linear head is chosen based on the task goal:
      - For 'classification': maps from n_feats to n_out.
      - For 'survival' or 'regression': maps from n_feats to 1.
      - For 'survival_discrete': maps from n_feats to n_out.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 goal: str = "classification"):
        super(linear_slide_classifier, self).__init__()
        self.goal = goal
        if goal == "classification":
            self.head = nn.Linear(n_feats, n_out)
        elif goal in ["survival", "regression"]:
            self.head = nn.Linear(n_feats, 1)
        elif goal == "survival_discrete":
            self.head = nn.Linear(n_feats, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class mlp_slide_classifier(nn.Module):
    """
    mlp_slide_classifier is a multi-layer perceptron for slide-level classification.
    
    This model consists of two parts:
      1) An encoder that transforms the input embedding into a latent representation.
      2) A prediction head that maps the latent representation to the output.
    
    The encoder is built with repeated blocks of Linear -> Activation -> BatchNorm1d -> Dropout.
    Using BatchNorm1d consistently ensures that the model weight keys remain stable.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 encoder_layers: int,
                 z_dim: int,
                 dropout_p: float = 0.5,
                 activation_function: Optional[str] = "ReLU",
                 goal: str = "classification"):
        super(mlp_slide_classifier, self).__init__()
        # Always use batch normalization on the feature dimension.
        self.encoder = build_slide_level_encoder(n_feats, z_dim, encoder_layers,
                                                 activation_function, dropout_p, use_batchnorm=True)
        if goal == "classification":
            self.head = nn.Linear(z_dim, n_out)
        elif goal in ["survival", "regression"]:
            self.head = nn.Linear(z_dim, 1)
        elif goal == "survival_discrete":
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        initialize_weights(self, activation_function)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, features).
        
        Returns:
            torch.Tensor: Output logits or predictions.
        """
        latent = self.encoder(x)
        logits = self.head(latent)
        return logits


# -----------------------------------------------------------------------------
# Script Test Section
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy parameters.
    batch_size = 4
    n_feats = 768
    n_out = 2

    # Create dummy slide-level embeddings (2D: [batch, features]).
    dummy_embeddings = torch.randn(batch_size, n_feats)

    # Test the linear_slide_classifier.
    lin_model = linear_slide_classifier(
        n_feats=n_feats,
        n_out=n_out,
        goal="classification"
    )
    logits_lin = lin_model(dummy_embeddings)
    print("linear_slide_classifier logits shape:", logits_lin.shape)

    # Test the mlp_slide_classifier.
    mlp_model = mlp_slide_classifier(
        n_feats=n_feats,
        n_out=n_out,
        encoder_layers=2,
        z_dim=256,
        dropout_p=0.5,
        activation_function="ReLU",
        goal="classification"
    )
    logits_mlp = mlp_model(dummy_embeddings)
    print("mlp_slide_classifier logits shape:", logits_mlp.shape)
