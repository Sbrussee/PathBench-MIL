"""
Here one can use pytorch modules as custom MIL aggregator models instead of the ones included in slideflow.
The construted modules can be imported into the benchmark.py script to be used in the benchmarking process.
As an example, we added some simple MIL methods (linear + mean, linear + max) below.

"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.mixture import GaussianMixture

def get_activation_function(activation_name: str):
    """Return the corresponding activation function from a string name."""
    if activation_name is None:
        return None
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def build_encoder(n_feats: int, z_dim: int, encoder_layers: int, activation_function: str, dropout_p: float = 0.1, use_batchnorm: bool = True):
    """Builds an encoder with a specified number of layers and activation functions."""
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

class linear_mil(nn.Module):
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

class mean_mil(nn.Module):
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
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, max_clusters: int = 10, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super(cluster_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        
        # Learnable cluster centroids
        self.cluster_centroids = nn.Parameter(torch.randn(max_clusters, z_dim))
        self.max_clusters = max_clusters
        
        # Final classifier head
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * max_clusters),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * max_clusters, n_out)
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
            
            # Compute distances to the cluster centroids
            distances = torch.cdist(embeddings, self.cluster_centroids)  # [n_patches, max_clusters]
            
            # Compute soft cluster assignments
            soft_assignments = F.softmax(-distances, dim=1)  # [n_patches, max_clusters]
            
            # Compute the embedding for each cluster as a weighted sum of patch embeddings
            cluster_embeddings = []
            for j in range(self.max_clusters):
                cluster_embedding = torch.sum(soft_assignments[:, j].unsqueeze(-1) * embeddings, dim=0)
                cluster_embeddings.append(cluster_embedding.unsqueeze(0))
            
            # Concatenate the cluster embeddings into a single vector
            cluster_embeddings = torch.cat(cluster_embeddings, dim=0)  # [max_clusters, z_dim]
            
            # Flatten the cluster embeddings to [z_dim * max_clusters]
            pooled_embeddings.append(cluster_embeddings.view(-1))  # [z_dim * max_clusters]

        # Stack all pooled embeddings and pass through the final head
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)  # [batch_size, z_dim * max_clusters]
        scores = self.head(pooled_embeddings)
        return scores

class gated_attention_mil(nn.Module):
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
    """Adaptive Instance Ranking MIL (AIR-MIL), which is a learnable top-k MIL model."""
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

class cair_mil(nn.Module):
    """Cross-correlation Adaptive Instance Ranking MIL (CAIR-MIL) with dynamic handling of single or multiple outputs."""
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, initial_k: int = 20, 
                 dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1,
                 nhead: int = 1, num_transformer_layers: int = 1, dim_feedforward: int = 512):
        super().__init__()
        self.n_out = n_out
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        
        # Transformer configuration for all instances
        transformer_layer = TransformerEncoderLayer(d_model=z_dim, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, dropout=dropout_p)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        # Attention mechanism to select top-k instances after the Transformer
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        
        # Attention pooling layer
        self.attention_pooling = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        
        # Separate heads or a single head based on the number of outputs
        if self.n_out > 1:
            self.output_layers = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        else:
            self.output_layer = nn.Linear(z_dim, n_out)
        
        # Learnable k
        self.k_param = nn.Parameter(torch.tensor(float(initial_k)))
        
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        
        # Pass all embeddings through the Transformer
        transformer_output = self.transformer(embeddings.transpose(0, 1)).transpose(0, 1)  # [batch_size, n_patches, z_dim]
        
        # Calculate attention scores based on Transformer output
        scores = self.attention(transformer_output).squeeze(-1)
        k = torch.clamp(self.k_param, 1, n_patches).int()
        
        # Select top-k embeddings based on the attention scores
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        topk_embeddings = torch.gather(transformer_output, 1, topk_indices.unsqueeze(-1).expand(-1, -1, transformer_output.size(-1)))
        
        # Apply softmax to top-k scores to get attention weights
        topk_attention_weights = F.softmax(topk_scores, dim=1)
        
        # Apply attention pooling to the top-k embeddings
        attention_scores = self.attention_pooling(topk_embeddings).squeeze(-1)  # [batch_size, k]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, k]
        pooled_output = torch.sum(topk_embeddings * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, z_dim]
        
        # Handle single or multiple outputs
        if self.n_out > 1:
            outputs = torch.stack([self.output_layers[i](pooled_output) for i in range(self.n_out)], dim=-1)
        else:
            outputs = self.output_layer(pooled_output)
            if outputs.shape[1] == 1:  # Binary classification case
                outputs = torch.cat([outputs, -outputs], dim=1)  # Convert to two logits
        
        if return_attention:
            return outputs, topk_attention_weights, attention_weights
        else:
            return outputs


class hierarchical_cluster_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, max_clusters: int = 10, dropout_p: float = 0.1, activation_function='ReLU',
                 encoder_layers=1):
        super().__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        
        # Learnable weights for clusters
        self.cluster_weights = nn.Parameter(torch.randn(z_dim, max_clusters))
        self.max_clusters = max_clusters
        
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
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        all_region_embeddings = []

        for i in range(batch_size):
            embeddings = self.encoder(bags[i].view(-1, n_feats))
            logits = embeddings @ self.cluster_weights.view(-1, 1).squeeze()  # [n_patches, max_clusters]
            cluster_probs = F.softmax(logits, dim=1)  # [n_patches, max_clusters]
            
            region_embeddings = cluster_probs.t() @ embeddings  # [max_clusters, z_dim]
            region_embeddings = self.region_head(region_embeddings)
            all_region_embeddings.append(region_embeddings)

        region_embeddings = torch.stack(all_region_embeddings, dim=0)  # [batch_size, max_clusters, z_dim]
        slide_attention_weights = F.softmax(self.slide_attention(region_embeddings), dim=1)
        slide_embedding = torch.sum(slide_attention_weights * region_embeddings, dim=1)
        scores = self.slide_head(slide_embedding)
        return scores

class adaptive_gmm_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, max_components: int = 10, 
                activation_function: str = "ReLU", dropout_p: float = 0.1, encoder_layers=1):
        super(adaptive_gmm_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function=activation_function, dropout_p=dropout_p, use_batchnorm=True)
        
        # Soft weights for each component
        self.component_weights = nn.Parameter(torch.ones(max_components))
        self.max_components = max_components
        
        # Final classifier head
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * max_components),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * max_components, n_out)
        )
        
        # Attention mechanism for component embeddings
        self.attention_layer = nn.Linear(z_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        all_bag_embeddings = []

        for i in range(batch_size):
            # Encode instances in the bag
            embeddings = self.encoder(bags[i].view(-1, n_feats))  # [n_patches, z_dim]
            
            # Fit GMM on the instance embeddings
            gmm = GaussianMixture(n_components=self.max_components, covariance_type='full', random_state=42)
            gmm.fit(embeddings.detach().cpu().numpy())
            
            # Get soft assignments of instances to Gaussian components
            posterior_probs = torch.tensor(gmm.predict_proba(embeddings.detach().cpu().numpy()), device=embeddings.device)
            
            # Learnable component weights
            component_weights = F.softmax(self.component_weights, dim=0)  # [max_components]
            
            # Compute weighted sum of embeddings for each component
            component_embeddings = []
            for j in range(self.max_components):
                weighted_sum = torch.sum(posterior_probs[:, j].unsqueeze(-1) * embeddings, dim=0)
                component_embeddings.append(weighted_sum * component_weights[j])
            
            # Stack component embeddings into a tensor
            component_embeddings = torch.cat(component_embeddings, dim=0)  # [max_components, z_dim]
            
            # Apply single-layer attention to the component embeddings
            attention_scores = self.attention_layer(component_embeddings.float()).squeeze(-1)  # [max_components]
            attention_weights = F.softmax(attention_scores, dim=0)  # [max_components]
            bag_embedding = torch.sum(attention_weights.unsqueeze(-1) * component_embeddings, dim=0)  # [z_dim]
            
            all_bag_embeddings.append(bag_embedding)
        
        # Stack all bag embeddings and pass through the final classifier head
        all_bag_embeddings = torch.stack(all_bag_embeddings, dim=0)
        scores = self.head(all_bag_embeddings)
        return scores


class dpp_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1,
                 activation_function='ReLU', encoder_layers=1):
        super(dpp_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.head = nn.Linear(z_dim, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)

        # Compute the kernel matrix
        kernel_matrix = torch.matmul(embeddings, embeddings.transpose(1, 2))
        
        # Add a small value to the diagonal for numerical stability
        kernel_matrix += torch.eye(kernel_matrix.size(1)).to(embeddings.device) * 1e-5
        
        # Compute determinant along the batch dimension
        kernel_det = kernel_matrix.det()  # Shape: (batch_size,)
        
        # Get top indices across patches, keeping the batch size intact
        _, top_indices = torch.topk(kernel_det, k=1, dim=1)  # Shape: (batch_size, 1)
        
        # Make sure top_indices is compatible for gathering
        top_indices = top_indices.unsqueeze(-1).expand(batch_size, 1, embeddings.size(-1))
        
        # Gather selected embeddings while preserving batch size
        selected_embeddings = torch.gather(embeddings, 1, top_indices)

        # Average the selected embeddings across the patch dimension
        pooled_embeddings = selected_embeddings.mean(dim=1)  # Shape: (batch_size, z_dim)
        
        # Compute final scores
        scores = self.head(pooled_embeddings)  # Shape: (batch_size, n_out)
        
        return scores

class il_mil(nn.Module):
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

class capsule_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, num_capsules=10, capsule_dim=16, 
                 routing_iters=3, dropout_p: float = 0.1, activation_function='ReLU', encoder_layers=1):
        super(capsule_mil, self).__init__()
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.capsule_layer = self.CapsuleLayer(num_capsules, n_out, z_dim, capsule_dim, routing_iters)
        self.classifier = nn.Linear(capsule_dim, n_out)
        self._initialize_weights()

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1).permute(0, 2, 1)
        capsules = self.capsule_layer(embeddings)
        bag_scores = self.classifier(capsules).mean(dim=-1)  # Adjust to match the required dimension
        return bag_scores

    def _initialize_weights(self):
        initialize_weights(self)

    class CapsuleLayer(nn.Module):
        def __init__(self, num_capsules, num_routes, in_channels, out_channels, routing_iters=3):
            super().__init__()
            self.num_routes = num_routes
            self.num_capsules = num_capsules
            self.routing_iters = routing_iters
            self.capsules = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, kernel_size=1) for _ in range(num_capsules)
            ])

        def forward(self, x):
            u = [capsule(x) for capsule in self.capsules]
            u = torch.stack(u, dim=2)
            u = u.transpose(1, 3)  # shape now (batch_size, num_routes, num_capsules, out_channels)
            b = torch.zeros(u.size(0), u.size(1), self.num_routes, 1, device=u.device)

            for i in range(self.routing_iters):
                c = F.softmax(b, dim=2)
                c = c.permute(0, 2, 1).unsqueeze(-1)  # Permute to match u's shape
                s = (c * u).sum(dim=2, keepdim=True)
                v = self.squash(s)
                if i < self.routing_iters - 1:
                    b = b + (u * v).sum(dim=-1, keepdim=True)

            return v.squeeze(-1)

        @staticmethod
        def squash(s, dim=-1):
            s_squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
            scale = s_squared_norm / (1 + s_squared_norm)
            return scale * s / torch.sqrt(s_squared_norm + 1e-8)

class prototype_attention_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, n_prototypes: int = 5, dropout_p: float = 0.1,
                 activation_function='ReLU', encoder_layers=1):
        super(prototype_attention_mil, self).__init__()
        self.n_out = n_out
        self.n_prototypes = n_prototypes
        
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.prototype_base = nn.Parameter(torch.randn(n_out, n_prototypes, z_dim))
        self.prototype_attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, n_prototypes)
        )
        self.attention = nn.Linear(z_dim, 1)
        self.classifier = nn.Linear(z_dim, n_out)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)

        dynamic_prototypes = []
        for i in range(self.n_out):
            proto_attention_weights = F.softmax(self.prototype_attention(embeddings.mean(dim=1)), dim=-1)  
            proto_base = self.prototype_base[i]  # (n_prototypes, z_dim)

            # Ensure proto_attention_weights and proto_base have compatible dimensions
            assert proto_attention_weights.shape[-1] == self.n_prototypes, f"Mismatch: {proto_attention_weights.shape[-1]} vs {self.n_prototypes}"
            assert proto_base.shape[0] == self.n_prototypes, f"Mismatch: {proto_base.shape[0]} vs {self.n_prototypes}"

            dynamic_proto = torch.einsum('bp...,pd->bd...', proto_attention_weights, proto_base)  # Result: (batch_size, z_dim)

            dynamic_prototypes.append(dynamic_proto)

        # Stack along a new dimension for consistent size
        dynamic_prototypes = torch.stack(dynamic_prototypes, dim=1)  # Shape: (batch_size, n_out, z_dim)

        proto_attention_scores = torch.einsum('bnd,bcd->bnc', embeddings, dynamic_prototypes.unsqueeze(0))  # Shape: (batch_size, n_out, n_prototypes)
        proto_attention_scores = proto_attention_scores.max(dim=1)[0]  
        proto_attention_scores = F.softmax(proto_attention_scores, dim=-1)

        bag_embeddings = torch.einsum('bn,bnc->bc', proto_attention_scores, dynamic_prototypes)  # Shape: (batch_size, z_dim)

        bag_scores = self.classifier(bag_embeddings)  # Shape: (batch_size, n_out)
        return bag_scores

class weighted_mean_mil(nn.Module):
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
        attention_weights = torch.cat(attention_weights_list, dim=1)

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

        attention_weights = torch.cat(attention_weights_list, dim=1)
        if apply_softmax:
            attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights


class clam_mil_mb(nn.Module):
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
            nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)]) for _ in range(n.branches)
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
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)