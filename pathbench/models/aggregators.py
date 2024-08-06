"""
Here one can use pytorch modules as custom MIL aggregator models instead of the ones included in slideflow.
The construted modules can be imported into the benchmark.py script to be used in the benchmarking process.
As an example, we added some simple MIL methods (linear + mean, linear + max) below.

"""

import torch
from torch import nn
import torch.nn.functional as F


class simple_linear_mil(nn.Module):
    """
    Simple Multiple instance learning model with linear layers, useful for linear evaluation.
    
    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network

    Methods
    -------
    forward(bags)
        Forward pass through the model
    """


    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Linear(n_feats, n_out)
        self.head = nn.Linear(n_out, n_out)

    def forward(self, bags):
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores
        

class linear_mil(nn.Module):
    """
    Multiple instance learning model with linear layers.
    
    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network

    Methods
    -------
    forward(bags)
        Forward pass through the model
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

class mean_mil(nn.Module):
    """
    Multiple instance learning model with mean pooling.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network

    Methods
    -------
    forward(bags)
        Forward pass through the model
    
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class max_mil(nn.Module):
    """
    Multiple instance learning model with max pooling.
    
        Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network

    Methods
    -------
    forward(bags)
        Forward pass through the model
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.max(dim=1)[0]
        scores = self.head(pooled_embeddings)
        return scores

class lse_mil(nn.Module):
    """
    Multiple instance learning model with log-sum-exp pooling.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    r : float
        scaling factor for log-sum-exp pooling
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network
    r : float
        scaling factor for log-sum-exp pooling

    Methods
    -------
    forward(bags)
        Forward pass through the model
    
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, r: float = 1.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.r = r

    def forward(self, bags):
        embeddings = self.encoder(bags)
        lse_pooling = self.r * torch.logsumexp(embeddings / self.r, dim=1)
        scores = self.head(lse_pooling)
        return scores


class lstm_mil(nn.Module):
    """
    Multiple instance learning model with LSTM-based pooling.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    lstm_dim : int
        Dimensionality of the LSTM hidden state
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    lstm : nn.LSTM
        LSTM network
    head : nn.Sequential
        Prediction head network
    

    Methods
    -------
    forward(bags)
        Forward pass through the model
    
    """

    use_lens=True

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, lstm_dim: int = 128, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(z_dim, lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(lstm_dim),
            nn.Dropout(dropout_p),
            nn.Linear(lstm_dim, n_out)
        )

    def forward(self, bags, lens):
        embeddings = self.encoder(bags)  # Shape: (batch_size, num_instances, z_dim)
        lens_cpu = lens.cpu().to(dtype=torch.int64)  # Ensure lengths are on CPU and of type int64 for packing
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lens_cpu, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embeddings)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Use the last hidden state of the LSTM as the pooled representation
        pooled_embeddings = hidden[-1]
        scores = self.head(pooled_embeddings)
        return scores

class deepset_mil(nn.Module):
    """Multiple instance learning model with DeepSet-based pooling.
    
    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    deepset_phi : nn.Sequential
        DeepSet phi network
    deepset_rho : nn.Sequential
        DeepSet rho network
    
    Methods
    -------
    forward(bags)
        Forward pass through the model
    
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.deepset_phi = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )
        self.deepset_rho = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        phi_output = self.deepset_phi(embeddings)
        pooled_embeddings = phi_output.sum(dim=1)
        scores = self.deepset_rho(pooled_embeddings)
        return scores


class distributionpooling_mil(nn.Module):
    """
    Multiple instance learning model with distribution pooling (mean and variance).
    
    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    head : nn.Sequential
        Prediction head network
    
    Methods
    -------
    forward(bags)
        Forward pass through the model
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        # Encoder to transform raw features into a higher-dimensional space
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        # Head network to produce final classification scores
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * 2),  # Since we'll concatenate mean and variance
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * 2, n_out)
        )

    def forward(self, bags):
        # Encode each instance in the bag
        embeddings = self.encoder(bags)
        # Calculate the mean and variance of the embeddings
        mean_embeddings = embeddings.mean(dim=1)
        variance_embeddings = embeddings.var(dim=1)
        # Concatenate the mean and variance
        pooled_embeddings = torch.cat([mean_embeddings, variance_embeddings], dim=1)
        # Pass through the head network to get final scores
        scores = self.head(pooled_embeddings)
        return scores

class dsmil(nn.Module):
    """
    Dual-Stream Multiple Instance Learning model with attention mechanism and Conv1D bag-level classifier.
    DSMIL first computes instance-level embeddings and scores, then aggregates them into bag-level embeddings,
    based on an attention mechanism. Finally, a Conv1D classifier is used to predict the bag-level label.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    instance_encoder : nn.Sequential
        Instance encoder network
    instance_classifier : nn.Linear
        Instance classifier network
    
    attention : nn.Sequential
        Attention network
    conv1d : nn.Conv1d
        Conv1D network
    bag_classifier : nn.Sequential
        Bag-level classifier network
    
    Methods
    -------
    forward(bags, return_attention)
        Forward pass through the model
    
    calculate_attention(bags, lens, apply_softmax)
        Calculate attention scores for the given bags
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.instance_classifier = nn.Linear(z_dim, 1)
        self.attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.conv1d = nn.Conv1d(z_dim, z_dim, kernel_size=1)
        self.bag_classifier = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags, return_attention=False):
        batch_size, num_instances, _ = bags.size()

        # Instance-level encoding and classification
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)

        # Max pooling stream
        max_score, max_idx = instance_scores.max(dim=1)
        critical_instance = bags[torch.arange(batch_size), max_idx]

        # Attention mechanism
        instance_embeddings = instance_features.view(batch_size, num_instances, -1)
        critical_embeddings = instance_embeddings[torch.arange(batch_size), max_idx]
        attention_weights = F.softmax(self.attention(instance_embeddings - critical_embeddings.unsqueeze(1)), dim=1)
        bag_embeddings = (attention_weights * instance_embeddings).sum(dim=1)

        # Bag-level classification
        bag_embeddings = bag_embeddings.unsqueeze(-1)
        conv_out = self.conv1d(bag_embeddings).squeeze(-1)
        scores = self.bag_classifier(conv_out)

        if return_attention:
            return scores, attention_weights
        else:
            return scores


    def calculate_attention(self, bags, lens, apply_softmax=None):
        batch_size, num_instances, _ = bags.size()
        
        # Instance-level encoding
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_embeddings = instance_features.view(batch_size, num_instances, -1)

        # Get critical instance
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        max_score, max_idx = instance_scores.max(dim=1)
        critical_embeddings = instance_embeddings[torch.arange(batch_size), max_idx]
        
        # Compute attention scores
        attention_scores = self.attention(instance_embeddings - critical_embeddings.unsqueeze(1))
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        
        return attention_scores


class varmil(nn.Module):
    """Multiple instance learning model with attention-based mean and variance pooling.
    
    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    attention : nn.Sequential
        Attention network
    head : nn.Sequential
        Prediction head network
    
    Methods
    -------
    forward(bags, return_attention)
        Forward pass through the model
        
    calculate_attention(bags, lens, apply_softmax)
        Calculate attention scores for the given bags
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim * 2),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * 2, n_out)
        )

    def forward(self, bags, return_attention=False):
        embeddings = self.encoder(bags)
        attention_weights = self.attention(embeddings).softmax(dim=1)
        mean_embeddings = (embeddings * attention_weights).sum(dim=1)
        var_embeddings = ((embeddings - mean_embeddings.unsqueeze(1)) ** 2 * attention_weights).sum(dim=1)
        aggregated_embeddings = torch.cat([mean_embeddings, var_embeddings], dim=1)
        scores = self.head(aggregated_embeddings)
        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def calculate_attention(self, bags, lens, apply_softmax=None):
        embeddings = self.encoder(bags)
        attention_scores = self.attention(embeddings)
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores