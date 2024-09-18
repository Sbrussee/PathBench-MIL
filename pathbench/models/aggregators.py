"""
Here one can use pytorch modules as custom MIL aggregator models instead of the ones included in slideflow.
The construted modules can be imported into the benchmark.py script to be used in the benchmarking process.

"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.mixture import GaussianMixture

def get_activation_function(activation_name: str):
    """Return the corresponding activation function from a string name.
    
    Args:
    - activation_name: Name of the activation function
    
    Returns:
    - activation_function: Activation function
    """
    if activation_name is None:
        return None
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def build_encoder(n_feats: int, z_dim: int, encoder_layers: int, activation_function: str, dropout_p: float = 0.1, use_batchnorm: bool = True):
    """Builds an encoder with a specified number of layers and activation functions.
    
    Args:
    - n_feats: Number of input features
    - z_dim: Dimension of the latent space
    - encoder_layers: Number of layers in the encoder
    - activation_function: Activation function to use in the encoder
    - dropout_p: Dropout probability
    - use_batchnorm: Whether to use batch normalization in the encoder

    Returns:
    - encoder: Encoder network
    """
    layers = []
    for i in range(encoder_layers):
        in_features = n_feats if i == 0 else z_dim
        layers.append(nn.Linear(in_features, z_dim))
        if activation_function is not None:
            layers.append(get_activation_function(activation_function))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(z_dim))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)

class linear_evaluation_mil(nn.Module):
    """
    Linear Evaluation MIL model. The model passes the instance embeddings through a linear layer and then
    through a classifier, while not using any pooling operation or activation function. Therefore useful
    for linear evaluation of the instance embeddings.
    
    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function=None, encoder_layers=1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            *[nn.Linear(n_feats if i == 0 else z_dim, z_dim) for i in range(encoder_layers)]
        )
        self.head = nn.Linear(z_dim, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

class mean_mil(nn.Module):
    """
    Mean-pooling MIL model. The model computes the mean pooling of the instance embeddings and passes the result
    through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

class max_mil(nn.Module):
    """
    Max-pooling MIL model. The model computes the max pooling of the instance embeddings and passes the result
    through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.max(dim=1)[0]
        scores = self.head(pooled_embeddings)
        return scores

class lse_mil(nn.Module):
    """
    Log-sum-exp MIL model. The model computes the log-sum-exp pooling of the instance embeddings and passes the result
    through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - r: Scaling factor for the log-sum-exp pooling
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 r: float = 1.0, activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.r = r
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        lse_pooling = self.r * torch.logsumexp(embeddings / self.r, dim=1)
        scores = self.head(lse_pooling)
        return scores

class lstm_mil(nn.Module):
    """
    LSTM MIL model. The model uses an LSTM to process the instance embeddings and passes the final hidden state
    through a classifier. 

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - lstm_dim: Dimension of the LSTM hidden state
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    use_lens = True

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, lstm_dim: int = 128, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.lstm = nn.LSTM(z_dim, lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(lstm_dim),
            nn.Dropout(dropout_p),
            nn.Linear(lstm_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, lens):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        lens_cpu = lens.cpu().to(dtype=torch.int64)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lens_cpu, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embeddings)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        pooled_embeddings = hidden[-1]
        scores = self.head(pooled_embeddings)
        return scores

class deepset_mil(nn.Module):
    """ Deep Sets MIL model. The model uses three fully connected layers to process the instance embeddings:
    the encoder network, phi and rho. The phi network processes the instance embeddings, which are summed and then the rho network
    classies the summed embeddings. 

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model
    
    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.deepset_phi = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            get_activation_function(activation_function),
            nn.Linear(z_dim, z_dim)
        )
        self.deepset_rho = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            get_activation_function(activation_function),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        phi_output = self.deepset_phi(embeddings)
        pooled_embeddings = phi_output.sum(dim=1)
        scores = self.deepset_rho(pooled_embeddings)
        return scores

class distributionpooling_mil(nn.Module):
    """
    Distribution Pooling MIL model. The model computes the mean and variance of the instance embeddings and
    concatenates these statistics. The model then passes the concatenated embeddings through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * 2),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * 2, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        mean_embeddings = embeddings.mean(dim=1)
        variance_embeddings = embeddings.var(dim=1)
        pooled_embeddings = torch.cat([mean_embeddings, variance_embeddings], dim=1)
        scores = self.head(pooled_embeddings)
        return scores

class dsmil(nn.Module):
    """
    Dual-stream MIL model. The model operates in two streams: instance-level and bag-level.
    The instance-level stream evaluates critical instances determined by the instance classifier, which is used
    to calculate the attention weights for each instance. The bag-level stream computes the bag embeddings
    using the attention-weighted instance embeddings. The bag embeddings are processed by a convolutional layer
    and passed through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.instance_encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, num_instances, _ = bags.size()
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_features = instance_features.view(batch_size, num_instances, -1)
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        max_score, max_idx = instance_scores.max(dim=1)
        critical_instance = bags[torch.arange(batch_size), max_idx]
        critical_embeddings = instance_features.view(batch_size, num_instances, -1)[torch.arange(batch_size), max_idx]
        attention_weights = F.softmax(self.attention(instance_features.view(batch_size, num_instances, -1) - critical_embeddings.unsqueeze(1)), dim=1)
        bag_embeddings = (attention_weights * instance_features.view(batch_size, num_instances, -1)).sum(dim=1)
        bag_embeddings = bag_embeddings.unsqueeze(-1)
        conv_out = self.conv1d(bag_embeddings).squeeze(-1)
        scores = self.bag_classifier(conv_out)

        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, num_instances, _ = bags.size()
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_embeddings = instance_features.view(batch_size, num_instances, -1)
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        max_score, max_idx = instance_scores.max(dim=1)
        critical_embeddings = instance_embeddings[torch.arange(batch_size), max_idx]
        attention_scores = self.attention(instance_embeddings - critical_embeddings.unsqueeze(1))
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores

class varmil(nn.Module):
    """
    Variance MIL model. The model computes the variance and mean of the attention-weighted instance embeddings. 
    The model then concatenates this mean and variance and passes the result through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model
    - calculate_attention: Calculate the attention weights for each instance

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_weights = self.attention(embeddings).softmax(dim=1)
        mean_embeddings = (embeddings * attention_weights).sum(dim=1)
        var_embeddings = ((embeddings - mean_embeddings.unsqueeze(1)) ** 2 * attention_weights).sum(dim=1)
        aggregated_embeddings = torch.cat([mean_embeddings, var_embeddings], dim=1)
        scores = self.head(aggregated_embeddings)
        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_scores = self.attention(embeddings)
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores

class perceiver_mil(nn.Module):
    """
    Perceiver MIL model. The model uses a learnable latent array to compute attention weights between the input features
    and the latent array. The model then applies transformer layers to the latent array and uses the CLS token for classification.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - latent_dim: Dimension of the latent array
    - num_latents: Number of learnable latents
    - num_layers: Number of transformer layers
    - num_heads: Number of attention heads
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - output: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, latent_dim: int = 128, 
                 num_latents: int = 16, num_layers: int = 6, num_heads: int = 8, 
                 dropout_p: float = 0.1, activation_function: str = 'ReLU', encoder_layers: int = 1):
        super(perceiver_mil, self).__init__()
        
        # Introduce a CLS token
        self.cls_token = nn.Parameter(torch.randn(1, latent_dim))
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout_p)
        
        # Transformer layers after cross-attention
        self.transformer_layers = TransformerEncoder(
            TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=z_dim, dropout=dropout_p),
            num_layers=num_layers
        )
        
        self.input_projection = nn.Linear(n_feats, latent_dim)
        self.output_projection = nn.Linear(latent_dim, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, attn_mask=None):
        batch_size, n_patches, n_feats = bags.shape
        
        # Project input features to latent dimension
        x = self.input_projection(bags.view(-1, n_feats)).view(batch_size, n_patches, -1)
        
        # Combine CLS token with latent space
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, latent_dim)
        latents = torch.cat([cls_tokens, self.latents.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)  # (batch_size, 1 + num_latents, latent_dim)
        
        # Cross-attention between input features and learnable latent array
        latents = latents.permute(1, 0, 2)  # (1 + num_latents, batch_size, latent_dim)
        x = x.permute(1, 0, 2)  # (n_patches, batch_size, latent_dim)
        latents, _ = self.cross_attention(latents, x, x)
        
        # Apply transformer layers to the latents
        latents = self.transformer_layers(latents)
        
        # Use the CLS token directly for classification
        cls_output = latents[0]  # (batch_size, latent_dim)
        output = self.output_projection(cls_output)  # (batch_size, n_out)
        return output

class cluster_mil(nn.Module):
    """
    Cluster MIL model. The model clusters the instance embeddings and computes the cluster centroids.
    The model then computes the distance between each instance and the cluster centroids and uses the distances
    to compute the instance weights. The model then weights the instance embeddings using soft cluster assignments
    and calculates attention scores for each instance using the weighted embeddings. The model then computes the
    cluster embeddings as a weighted sum of the instance embeddings and then calculates attention scores
    for each cluster. The model then computes the final bag embedding as an attention-weighted sum of the cluster embeddings
    and passes the result through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - max_clusters: Maximum number of clusters
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag. 
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, max_clusters: int = 10, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super(cluster_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        
        # Learnable cluster centroids
        self.cluster_centroids = nn.Parameter(torch.randn(max_clusters, z_dim))
        self.max_clusters = max_clusters

        self.instance_attention = Attention(z_dim)
        self.cluster_attention = Attention(z_dim)
        
        # Final classifier head
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * max_clusters),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * max_clusters, n_out)
        )
        self._initialize_weights()


    def _initialize_weights(self):
        initialize_weights(self)

    class Attention(nn.Module):
        def __init__(self, in_dim):
            super(cluster_mil.Attention, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.Tanh(),
                nn.Linear(in_dim, 1)
            )
        
        def forward(self, x):
            # x: [n_instances, in_dim] or [n_clusters, in_dim]
            weights = self.attention(x)  # [n_instances, 1] or [n_clusters, 1]
            weights = F.softmax(weights, dim=0)  # Apply softmax over instances or clusters
            return weights

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        pooled_embeddings = []

        for i in range(batch_size):
            # Encode the patches
            embeddings = self.encoder(bags[i].view(-1, n_feats))  # [n_patches, z_dim]
            
            # Compute distances to the cluster centroids
            distances = torch.cdist(embeddings, self.cluster_centroids)  # [n_patches, max_clusters]
            
            # Compute soft cluster assignments
            soft_assignments = F.softmax(-distances, dim=1)  # [n_patches, max_clusters]
            
            # Compute the embedding for each cluster as a weighted sum of patch embeddings
            cluster_embeddings = []
            for j in range(self.max_clusters):
                weighted_embeddings = soft_assignments[:, j].unsqueeze(-1) * embeddings  # [n_patches, z_dim]
                attention_weights = self.instance_attention(weighted_embeddings)  # [n_patches, 1]
                cluster_embedding = torch.sum(attention_weights * weighted_embeddings, dim=0)  # [z_dim]
                cluster_embeddings.append(cluster_embedding.unsqueeze(0))
            
            # Concatenate the cluster embeddings into a single vector
            cluster_embeddings = torch.cat(cluster_embeddings, dim=0)  # [max_clusters, z_dim]
            
            cluster_attention = self.cluster_attention(cluster_embeddings)  # [max_clusters, 1]
            cluster_sum = torch.sum(cluster_attention * cluster_embeddings, dim=0)  # [z_dim]
            pooled_embeddings.append(cluster_sum.unsqueeze(0))

        # Stack all pooled embeddings and pass through the final head
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # [batch_size, z_dim * max_clusters]
        scores = self.head(pooled_embeddings)
        return scores

class gated_attention_mil(nn.Module):
    """
    Gated Attention MIL model. The model computes attention weights for each instance and aggregates the instance
    embeddings using the attention weights in a gated manner. The model then passes the aggregated embeddings through a classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1) -> None:
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_V = self.attention_V(embeddings)
        attention_U = self.attention_U(embeddings)
        attention_weights = self.attention_weights(attention_V * attention_U).softmax(dim=1)
        pooled_embeddings = (embeddings * attention_weights).sum(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_V = self.attention_V(embeddings)
        attention_U = self.attention_U(embeddings)
        attention_weights = self.attention_weights(attention_V * attention_U)
        if apply_softmax:
            attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights

class topk_mil(nn.Module):
    """
    Top-k MIL model. The model computes attention weights for each instance and selects the top-k instances
    based on these weights. The model then computes the weighted sum of the top-k instances and passes the result
    through a classifier. Useful for problems where the number of relevant instances can be estimated.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space

    Methods:
    - forward: Forward pass through the model
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, k: int = 20, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.k = k
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)
        
        k = min(self.k, n_patches)
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        topk_embeddings = torch.gather(embeddings, 1, topk_indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1)))
        
        # Apply softmax to top-k scores to get attention weights
        topk_attention_weights = F.softmax(topk_scores, dim=1)
        
        # Compute the weighted sum of the top-k embeddings
        weighted_sum = torch.sum(topk_embeddings * topk_attention_weights.unsqueeze(-1), dim=1)
        
        if return_attention:
            return self.head(weighted_sum), topk_attention_weights
        else:
            return self.head(weighted_sum)

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores = F.softmax(scores, dim=1)
        return scores

class air_mil(nn.Module):
    """Adaptive Instance Ranking MIL (AIR-MIL), which is a learnable top-k MIL model. First, the model computes
    attention weights for each instance and selects the top-k instances based on these weights. The model then
    computes the weighted mean of the top-k instances and passes the result through a classifier. The model also
    includes a learnable parameter k that determines the number of instances to select. Useful for problems
    where the number of relevant intances is unknown.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - initial_k: Initial value of the learnable parameter k
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, initial_k: int = 20, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        
        # Always learnable k
        self.k_param = nn.Parameter(torch.tensor(float(initial_k)), requires_grad=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)

        # Bound k to be between 1 and n_patches
        k = torch.clamp(self.k_param, 1, n_patches).int()
        
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        topk_embeddings = torch.gather(embeddings, 1, topk_indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1)))
        
        # Apply softmax to top-k scores to get attention weights
        topk_attention_weights = F.softmax(topk_scores, dim=1)
        
        # Compute the weighted mean of the top-k embeddings
        weighted_sum = torch.sum(topk_embeddings * topk_attention_weights.unsqueeze(-1), dim=1)
        weighted_mean = weighted_sum / torch.sum(topk_attention_weights, dim=1, keepdim=True)
        
        if return_attention:
            return self.head(weighted_mean), topk_attention_weights
        else:
            return self.head(weighted_mean)

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores = F.softmax(scores, dim=1)
        return scores



class il_mil(nn.Module):
    """
    Instance-level MIL model. The model classifies each instance and aggregates the instance predictions
    using the mean operation. Useful for cases where a small number of relevant instances are present in each bag.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - bag_scores: Predicted class scores for in the bag
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, 
                 activation_function='ReLU', encoder_layers=1):
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.instance_classifier = nn.Linear(z_dim, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        # Flatten bags to pass through the encoder
        embeddings = self.encoder(bags.view(-1, n_feats))
        # Reshape to original bag structure
        embeddings = embeddings.view(batch_size, n_patches, -1)
        # Classify each instance
        instance_scores = self.instance_classifier(embeddings)
        # Aggregate instance predictions (mean)
        bag_scores = instance_scores.mean(dim=1)
        return bag_scores

class weighted_mean_mil(nn.Module):
    """
    Weighted mean MIL model. The variance of the instance embeddings is used to compute instance weights.
    The instance embeddings are then aggregated using the computed weights. Useful for noisy data.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Returns:
    - output: Predicted class scores for in the bag
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU',
    encoder_layers=1):
        super(weighted_mean_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        variances = embeddings.var(dim=1, unbiased=False)
        weights = 1.0 / (variances + 1e-5)
        weighted_sum = torch.sum(weights.unsqueeze(1) * embeddings, dim=1)
        robust_mean = weighted_sum / torch.sum(weights, dim=1, keepdim=True)
        output = self.head(robust_mean)
        return output


class clam_mil(nn.Module):
    """
    Clusering-constrained Attention MIL (CLAM-MIL) model. The model uses a learnable attention mechanism to
    aggregate instance embeddings into slide-level representations. The model also includes a clustering module
    to predict instance-level cluster assignments. For each output class the model has a separate attention
    mechanism and classifier. 

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - cluster_patches: Cluster patches into clusters
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    - attention_weights: Attention weights for each instance (optional)
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU',
    encoder_layers=1):
        super(clam_mil, self).__init__()
        self.n_out = n_out
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.attention_U = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh()
        )
        self.attention_branches = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        self.classifiers = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        self.instance_cluster = nn.ModuleList([nn.Linear(z_dim, 2) for _ in range(n_out)])
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        slide_level_representations = []
        attention_weights_list = []

        for i in range(self.n_out):
            attention_U_output = self.attention_U(embeddings)
            attention_V_output = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i](attention_U_output * attention_V_output).softmax(dim=1)
            attention_weights_list.append(attention_scores)
            slide_level_representation = torch.sum(attention_scores * embeddings, dim=1)
            slide_level_representations.append(slide_level_representation)

        slide_level_representations = torch.stack(slide_level_representations, dim=1)
        scores = []
        for i in range(self.n_out):
            score = self.classifiers[i](slide_level_representations[:, i, :])
            scores.append(score)
        scores = torch.cat(scores, dim=1)

        # Stack the attention weights and then take the mean along the first dimension (across branches)
        attention_weights = torch.stack(attention_weights_list, dim=0)  # Shape: (n_out, batch_size, n_patches, 1)
        attention_weights = torch.mean(attention_weights, dim=0)  # Shape: (batch_size, n_patches, 1)

        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def cluster_patches(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        cluster_predictions = []
        for i in range(self.n_out):
            cluster_pred = self.instance_cluster[i](embeddings)
            cluster_predictions.append(cluster_pred)
        return cluster_predictions

    def calculate_attention(self, bags, lens=None, apply_softmax=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_weights_list = []

        for i in range(self.n_out):
            attention_U_output = self.attention_U(embeddings)
            attention_V_output = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i](attention_U_output * attention_V_output).softmax(dim=1)
            attention_weights_list.append(attention_scores)

        attention_weights = torch.mean(torch.stack(attention_weights_list), dim=0)
        if apply_softmax:
            attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights


class clam_mil_mb(nn.Module):
    """
    CLAM-MIL model with multiple branches, each with its own attention mechanism and classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - n_branches: Number of branches
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for instances in the bag
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, n_branches: int = 3, activation_function='ReLU',
    encoder_layers=1):
        super(clam_mil_mb, self).__init__()
        self.n_out = n_out
        self.n_branches = n_branches
        self.encoders = nn.ModuleList([
            build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True) for _ in range(n_branches)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Sigmoid()
            ) for _ in range(n_branches)
        ])
        self.attention_V = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Tanh()
            ) for _ in range(n_branches)
        ])
        self.attention_branches = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)]) for _ in range(n_branches)
        ])
        self.classifiers = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)]) for _ in range(n_branches)
        ])
        self.instance_cluster = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 2) for _ in range(n_out)]) for _ in range(n.branches)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        
        all_scores = []
        all_attention_weights = []
        
        # Process each branch separately
        for branch_idx in range(self.n_branches):
            embeddings = self.encoder[bags.view(-1, n_feats)]
            embeddings = embeddings.view(batch_size, n_patches, -1)
            
            slide_level_representations = []
            attention_weights_list = []

            for i in range(self.n_out):
                attention_U_output = self.attention_U[branch_idx](embeddings)
                attention_V_output = self.attention_V[branch_idx](embeddings)
                attention_scores = self.attention_branches[branch_idx][i](attention_U_output * attention_V_output).softmax(dim=1)
                attention_weights_list.append(attention_scores)
                slide_level_representation = torch.sum(attention_scores * embeddings, dim=1)
                slide_level_representations.append(slide_level_representation)

            slide_level_representations = torch.stack(slide_level_representations, dim=1)
            scores = []
            for i in range(self.n_out):
                score = self.classifiers[branch_idx][i](slide_level_representations[:, i, :])
                scores.append(score)
            scores = torch.cat(scores, dim=1)
            attention_weights = torch.cat(attention_weights_list, dim=1)

            all_scores.append(scores)
            all_attention_weights.append(attention_weights)

        # Combine the results from all branches
        combined_scores = torch.mean(torch.stack(all_scores), dim=0)
        combined_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
        
        if return_attention:
            return combined_scores, combined_attention_weights
        else:
            return combined_scores

    def calculate_attention(self, bags, lens=None, apply_softmax=False):
        batch_size, n_patches, n_feats = bags.shape
        
        all_attention_weights = []

        # Process each branch separately
        for branch_idx in range(self.n_branches):
            embeddings = self.encoder[bags.view(-1, n_feats)]
            embeddings = embeddings.view(batch_size, n_patches, -1)
            
            attention_weights_list = []

            for i in range(self.n_out):
                attention_U_output = self.attention_U[branch_idx](embeddings)
                attention_V_output = self.attention_V[branch_idx](embeddings)
                attention_scores = self.attention_branches[branch_idx][i](attention_U_output * attention_V_output)
                if apply_softmax:
                    attention_scores = F.softmax(attention_scores, dim=1)
                attention_weights_list.append(attention_scores)

            attention_weights = torch.cat(attention_weights_list, dim=1)
            all_attention_weights.append(attention_weights)

        # Combine the attention weights from all branches
        combined_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
        return combined_attention_weights


def initialize_weights(module):
    """
    Initialize the weights of the model using Xavier initialization for linear layers and constant initialization
    for batch normalization layers.

    Args:
    - module: The model to initialize

    Returns:
    - None
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class gmm_mil(nn.Module):
    """
    Gaussian Mixture Model MIL model, where each bag is represented as a pooled embedding from a GMM.

    We fit a GMM to the patch embeddings, and then create a pooled embedding by concatenating the means
    and covariances of the GMM components.

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Args:
    - n_feats: Number of input features (e.g., patch dimensionality)
    - n_out: Number of output classes
    - z_dim: Dimensionality of the patch embeddings
    - num_components: Number of components in the GMM
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Returns:
    - scores: Predicted class scores for each bag in the batch.
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, num_components: int = 10, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super(gmm_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, False) # No batch norm since we iterate over the batches
        self.gmm = GaussianMixture(n_components=num_components, covariance_type='full')
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * num_components),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * num_components, n_out)
        )
        self.num_components = num_components
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        pooled_embeddings = []

        for i in range(batch_size):
            # Encode the patches
            embeddings = self.encoder(bags[i].view(-1, n_feats))  # [n_patches, z_dim]

            # Fit GMM to the embeddings
            self.gmm.fit(embeddings.detach().cpu().numpy())
            gmm_means = torch.tensor(self.gmm.means_, device=embeddings.device)  # [num_components, z_dim]
            gmm_covariances = torch.tensor(self.gmm.covariances_, device=embeddings.device)  # [num_components, z_dim, z_dim]
            
            # Flatten means and covariances to create a pooled embedding
            pooled_embedding = torch.cat([gmm_means.view(-1), gmm_covariances.view(-1)], dim=0)  # [z_dim * num_components * 2]
            pooled_embeddings.append(pooled_embedding.unsqueeze(0))

        # Stack all pooled embeddings and pass through the final head
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # [batch_size, z_dim * num_components * 2]
        scores = self.head(pooled_embeddings)
        return scores

class prototype_mil(nn.Module):
    """
    Prototype-based MIL model, where each bag is represented as a weighted sum of prototype embeddings.

    We define a set of learnable prototypes, and compute the prototype assignments for each patch in the bag.
    The bag embedding is then computed as a weighted sum of the patch embeddings, where the weights are the
    prototype assignments.

    Methods:
    - forward: Forward pass through the model
    - initialize_weights: Initialize the weights of the model

    Args:
    - n_feats: Number of input features (e.g., patch dimensionality)
    - n_out: Number of output classes
    - z_dim: Dimensionality of the patch embeddings
    - num_prototypes: Number of learnable prototypes
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Returns:
    - scores: Predicted class scores for each bag in the batch.
    """
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, num_prototypes: int = 10, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super(prototype_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, False) # No batch norm since we iterate over the batches
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, z_dim))
        self.num_prototypes = num_prototypes

        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * num_prototypes),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * num_prototypes, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        pooled_embeddings = []

        for i in range(batch_size):
            # Encode the patches
            embeddings = self.encoder(bags[i].view(-1, n_feats))  # [n_patches, z_dim]

            # Compute distances to the prototypes
            distances = torch.cdist(embeddings, self.prototypes)  # [n_patches, num_prototypes]

            # Compute soft prototype assignments
            soft_assignments = F.softmax(-distances, dim=1)  # [n_patches, num_prototypes]

            # Compute the embedding for each prototype as a weighted sum of patch embeddings
            prototype_embeddings = []
            for j in range(self.num_prototypes):
                prototype_embedding = torch.sum(soft_assignments[:, j].unsqueeze(-1) * embeddings, dim=0)
                prototype_embeddings.append(prototype_embedding.unsqueeze(0))

            # Concatenate the prototype embeddings into a single vector
            prototype_embeddings = torch.cat(prototype_embeddings, dim=0)  # [num_prototypes, z_dim]
            pooled_embedding = prototype_embeddings.view(-1)  # [z_dim * num_prototypes]
            pooled_embeddings.append(pooled_embedding.unsqueeze(0))

        # Stack all pooled embeddings and pass through the final head
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # [batch_size, z_dim * num_prototypes]
        scores = self.head(pooled_embeddings)
        return scores

class capsule_mil(nn.Module):
    """
    Capsule-based MIL model, inspired by the Capsule Network architecture: https://arxiv.org/abs/1710.09829

    The model consists of two main components:
    - Primary Capsule Layer: A set of convolutional capsules that extract features from the input patches 
    - Dynamic Routing Layer: A dynamic routing mechanism that iteratively refines the capsule outputs

    The final bag-level prediction is made by a linear classifier on the output of the digit capsules.

    Methods:
    - squash: Applies the squashing function to the input tensor
    - forward: Forward pass through

    Args:
    - n_feats: Number of input features (e.g., patch dimensionality)
    - n_out: Number of output classes
    - n_primary_capsules: Number of primary capsules
    - primary_out_dim: Output dimensionality of the primary capsules

    - n_digit_capsules: Number of digit capsules
    - digit_out_dim: Output dimensionality of the digit capsules
    - routing_iterations: Number of routing iterations in the dynamic routing

    Returns:
    - scores: Predicted class scores for each bag in the batch
    """

    def __init__(self, n_feats: int, n_out: int, n_primary_capsules: int = 8, primary_out_dim: int = 32, 
                 n_digit_capsules: int = 16, digit_out_dim: int = 16, routing_iterations: int = 3):
        super(capsule_mil, self).__init__()
        
        # Primary Capsule Layer
        self.n_primary_capsules = n_primary_capsules
        self.primary_capsules = nn.ModuleList([
            nn.Conv1d(n_feats, primary_out_dim, kernel_size=3, stride=1, padding=1)
            for _ in range(n_primary_capsules)
        ])
        
        # Dynamic Routing Layer
        self.n_digit_capsules = n_digit_capsules
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(n_digit_capsules, primary_out_dim * n_primary_capsules, digit_out_dim))
        
        # Final Classifier
        self.classifier = nn.Linear(digit_out_dim * n_digit_capsules, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        
        # Primary Capsule Layer
        u = [capsule(bags.view(batch_size * n_patches, n_feats, 1)).squeeze(-1) for capsule in self.primary_capsules]
        u = torch.stack(u, dim=-1)  # Shape: (batch_size * n_patches, primary_out_dim, n_primary_capsules)
        u = u.view(batch_size, n_patches, -1)  # Reshape to (batch_size, n_patches, primary_out_dim * n_primary_capsules)
        u = self.squash(u)  # Apply the squashing function

        # Dynamic Routing Layer
        b = torch.zeros(batch_size, n_patches, self.n_digit_capsules, device=bags.device)  # Routing logits
        for i in range(self.routing_iterations):
            c = F.softmax(b, dim=2)  # Routing weights
            s = (c.unsqueeze(-1) * u.unsqueeze(2)).matmul(self.W)  # Weighted sum
            v = self.squash(s.sum(dim=1))  # Apply squashing function
            if i < self.routing_iterations - 1:
                b = b + (u.unsqueeze(2) * v.unsqueeze(1)).sum(dim=-1)  # Update routing logits

        # Final Classifier
        v = v.view(batch_size, -1)  # Flatten the capsule output
        scores = self.classifier(v)  # Bag-level prediction
        
        return scores

    @staticmethod
    def squash(s):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + 1e-8))