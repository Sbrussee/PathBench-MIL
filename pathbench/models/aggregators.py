"""
Here one can use pytorch modules as custom MIL aggregator models instead of the ones included in slideflow.
The construted modules can be imported into the benchmark.py script to be used in the benchmarking process.
As an example, we added some simple MIL methods (linear + mean, linear + max) below.

"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


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

class cluster_pooling_mil(nn.Module):
    """
    Multiple instance learning model with cluster-based pooling. It first clusters patches into regions, pools each region using mean pooling
    and then aggregates region embeddings into the slide-level label using a linear classifier.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    n_clusters : int
        Number of clusters
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    kmeans : KMeans
        KMeans clustering model
    head : nn.Sequential
        Prediction head network
    
    Methods
    -------
    forward(bags)
        Forward pass through the model
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, n_clusters: int = 4, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * n_clusters),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * n_clusters, n_out)
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, bags):
        embeddings = self.encoder(bags)
        clusters = self.kmeans.fit_predict(embeddings.detach().cpu().numpy())
        pooled_embeddings = torch.cat([embeddings[clusters == i].mean(dim=0) for i in range(self.kmeans.n_clusters)], dim=0)
        pooled_embeddings = pooled_embeddings.view(1, -1)
        scores = self.head(pooled_embeddings)
        return scores

class gated_attention_mil(nn.Module):
    """
    Multiple instance learning model with gated attention mechanism.

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
    attention_V : nn.Sequential
        Attention network
    attention_U : nn.Sequential
        Attention network
    attention_weights : nn.Linear
        Attention network
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
        self.attention_V = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(z_dim, 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )


    def forward(self, bags):
        embeddings = self.encoder(bags)
        attention_V = self.attention_V(embeddings)
        attention_U = self.attention_U(embeddings)
        attention_weights = self.attention_weights(attention_V * attention_U).softmax(dim=1)
        pooled_embeddings = (embeddings * attention_weights).sum(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

class topk_mil(nn.Module):
    """
    Multiple instance learning model which selects the top k instances based on attention scores and aggregates them using mean pooling.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    k : int
        Number of top instances to select
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
    k : int
        Number of top instances to select
    
    Methods
    -------
    forward(bags)
        Forward pass through the model

    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, k: int = 20, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.k = k

    def forward(self, bags):
        embeddings = self.encoder(bags)
        scores = self.attention(embeddings).squeeze(-1)
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=0)
        topk_embeddings = embeddings[topk_indices]
        pooled_embeddings = topk_embeddings.mean(dim=0)
        scores = self.head(pooled_embeddings)
        return scores

class hierarchical_mil(nn.Module):
    """
    Hierarchical Multiple Instance Learning model with attention mechanism. It first groups consecutive patches into regions,
    weights instances per region using attention and then aggregates region embeddings into slide embeddings using an attention mechanism.
    Finally, it predicts the slide-level label using a linear classifier.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    region_size : int
        Number of patches per region
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    region_attention : nn.Sequential
        Region attention network
    region_head : nn.Sequential
        Region head network
    slide_attention : nn.Sequential
        Slide attention network
    slide_head : nn.Sequential
        Slide head network

    Methods
    -------
    forward(bags)
        Forward pass through the model

    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, region_size: int = 4, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.region_attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.region_head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, z_dim)
        )
        self.slide_attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.slide_head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.region_size = region_size

    def forward(self, bags):
        # Split into regions
        n_regions = bags.size(0) // self.region_size
        embeddings = self.encoder(bags)
        region_embeddings = []

        for i in range(n_regions):
            region_patches = embeddings[i * self.region_size: (i + 1) * self.region_size]
            attention_weights = F.softmax(self.region_attention(region_patches), dim=0)
            region_embedding = torch.sum(attention_weights * region_patches, dim=0)
            region_embeddings.append(self.region_head(region_embedding))
        
        region_embeddings = torch.stack(region_embeddings, dim=0)

        # Attention pooling across regions
        region_attention_weights = F.softmax(self.slide_attention(region_embeddings), dim=0)
        slide_embedding = torch.sum(region_attention_weights * region_embeddings, dim=0)

        # Final slide-level prediction
        scores = self.slide_head(slide_embedding)
        return scores

class hierarchical_cluster_mil(nn.Module):
    """
    Inspired by "Cluster-to-Conquer: A Framework for End-to-End
Multi-Instance Learning for Whole Slide Image Classification".

    Hierarchical Cluster MIL model with attention mechanism. It first clusters patches into regions, weights instances
    per region using attention and then aggregates region embeddings into slide embeddings using an attention mechanism. Finally, it predicts the
    slide-level label using a linear classifier.

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    n_clusters : int
        Number of clusters
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network
    kmeans : KMeans
        KMeans clustering model
    region_attention : nn.Sequential
        Region attention network
    region_head : nn.Sequential
        Region head network
    slide_attention : nn.Sequential
        Slide attention network
    slide_head : nn.Sequential
        Slide head network
    
    Methods
    -------
    forward(bags)
        Forward pass through the model
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, n_clusters: int = 10, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.region_attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.region_head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, z_dim)
        )
        self.slide_attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.slide_head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        # Encode each patch
        embeddings = self.encoder(bags)

        # Cluster patches into regions
        clusters = self.kmeans.fit_predict(embeddings.detach().cpu().numpy())

        region_embeddings = []

        for i in range(self.kmeans.n_clusters):
            cluster_patches = embeddings[clusters == i]
            attention_weights = F.softmax(self.region_attention(cluster_patches), dim=0)
            region_embedding = torch.sum(attention_weights * cluster_patches, dim=0)
            region_embeddings.append(self.region_head(region_embedding))
        
        region_embeddings = torch.stack(region_embeddings, dim=0)

        # Attention pooling across regions
        region_attention_weights = F.softmax(self.slide_attention(region_embeddings), dim=0)
        slide_embedding = torch.sum(region_attention_weights * region_embeddings, dim=0)

        # Final slide-level prediction
        scores = self.slide_head(slide_embedding)
        return scores

class retmil(nn.Module):
    """
    Retention-based MIL. Retention mechanism is applied to both local subsequences and the global sequence in a hierarchical manner.
    The local retention mechanism uses relative distance decay to compute the retention scores, while the global retention mechanism
    uses self-attention to compute the retention scores. The final prediction is made using a linear classifier.
    Method from: "RetMIL: Retentive Multiple Instance Learning
    for Histopathological Whole Slide Image
    Classification".

    Parameters
    ----------
    n_feats : int
        Number of input features
    n_out : int
        Number of output classes
    z_dim : int
        Dimensionality of the hidden layer
    subseq_len : int
        Length of subsequences
    n_heads : int
        Number of attention heads
    dropout_p : float
        Dropout probability
    
    Attributes
    ----------
    local_retention_1 : RetentionLayer
        Local retention mechanism 1
    local_retention_2 : RetentionLayer
        Local retention mechanism 2
    local_attention : AttentionPooling
        Local attention pooling
    global_retention : RetentionLayer
        Global retention mechanism
    global_attention : AttentionPooling
        Global attention pooling
    classifier : nn.Sequential
        Prediction head network
    
    Methods
    -------
    forward(x)
        Forward pass through the model
    
    Classes
    -------
    RetentionLayer(nn.Module)
        Retention mechanism for local and global sequences
    AttentionPooling(nn.Module)
        Attention pooling mechanism
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, subseq_len: int = 512, n_heads: int = 8, dropout_p: float = 0.1):
        super(retmil, self).__init__()
        self.subseq_len = subseq_len
        self.n_heads = n_heads

        # Retention mechanism for local subsequences
        self.local_retention_1 = self.RetentionLayer(z_dim, n_heads)
        self.local_retention_2 = self.RetentionLayer(z_dim, n_heads)
        self.local_attention = self.AttentionPooling(z_dim)

        # Retention mechanism for global sequence
        self.global_retention = self.RetentionLayer(z_dim, n_heads)
        self.global_attention = self.AttentionPooling(z_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        # Step 1: Split into subsequences
        subsequences = torch.split(bags, self.subseq_len)

        # Step 2: Local subsequence processing
        local_embeddings = []
        for subseq in subsequences:
            subseq = self.local_retention_1(subseq)
            subseq = self.local_retention_2(subseq)
            local_embedding = self.local_attention(subseq)
            local_embeddings.append(local_embedding)

        # Stack local embeddings into a global sequence
        global_sequence = torch.stack(local_embeddings)

        # Step 3: Global sequence processing
        global_sequence = self.global_retention(global_sequence)
        global_embedding = self.global_attention(global_sequence)

        # Step 4: Classification
        out = self.classifier(global_embedding)
        return out

    class RetentionLayer(nn.Module):
        def __init__(self, d_model, n_heads):
            super(RetMIL.RetentionLayer, self).__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            self.WQ = nn.Linear(d_model, d_model)
            self.WK = nn.Linear(d_model, d_model)
            self.WV = nn.Linear(d_model, d_model)

            self.group_norm = nn.GroupNorm(1, d_model)

        def forward(self, x):
            B, N, D = x.shape

            Q = self.WQ(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.WK(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.WV(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

            # Implementing a simplified version of retention mechanism
            # Retention using relative distance decay
            D = self.relative_distance_decay(N, K.device)
            retention_scores = (Q @ K.transpose(-2, -1)) * D
            retention_out = (retention_scores @ V).transpose(1, 2).contiguous().view(B, N, D)

            # Applying normalization
            retention_out = self.group_norm(retention_out)

            return retention_out

        def relative_distance_decay(self, N, device):
            D = torch.zeros(N, N, device=device)
            for i in range(N):
                for j in range(i, N):
                    D[i, j] = torch.exp(-float(j - i))
            return D

    class AttentionPooling(nn.Module):
        def __init__(self, d_model):
            super(RetMIL.AttentionPooling, self).__init__()
            self.W = nn.Linear(d_model, d_model)
            self.U = nn.Linear(d_model, d_model)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            alpha = F.softmax(self.gamma * torch.tanh(self.W(x)) * torch.sigmoid(self.U(x)), dim=1)
            out = torch.sum(alpha * x, dim=1)
            return out

class weighted_mean_mil(nn.Module):
    """
    Multiple instance learning model with weighted mean pooling. Instances are inversely weighted by their variance.
    This effectively makes the model more robust to noisy instances.

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
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1):
        super(weighted_mean_mil, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        
        # Compute robust mean (weighted mean with inverse variance)
        variances = embeddings.var(dim=0, unbiased=False)
        weights = 1.0 / (variances + 1e-5)  # Small epsilon to avoid division by zero
        weighted_sum = torch.sum(weights * embeddings, dim=0)
        robust_mean = weighted_sum / torch.sum(weights)

        output = self.head(robust_mean)
        return output

class aodmil(nn.Module):
    """
    Attention-based Outlier Detection Multiple Instance Learning model. The model computes the mean embedding of the bag,
    computes the similarity of each instance to the mean embedding, and computes the attention weights based on the similarity.
    The final prediction is made using a linear classifier.
    The method aims to detect patches that show abnormal behavior with respect to the other patches in the bag.

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
    forward(bags)
        Forward pass through the model
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1):
        super(aodmil, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )

    def forward(self, bags):
        embeddings = self.encoder(bags)
        
        # Compute the mean embedding of the bag
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        
        # Compute similarity of each instance to the mean embedding
        similarities = F.cosine_similarity(embeddings, mean_embedding, dim=-1)
        
        # Inverse attention based on similarity
        attention_weights = 1.0 - similarities
        attention_weights = self.attention(attention_weights.unsqueeze(-1)).squeeze(-1)
        
        # Weighted aggregation of embeddings
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)
        aggregated_embedding = weighted_embeddings.mean(dim=0)
        
        output = self.head(aggregated_embedding)
        return output