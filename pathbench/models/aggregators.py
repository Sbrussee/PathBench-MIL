"""
Here one can use pytorch modules as custom MIL aggregator models instead of the ones included in slideflow.
The construted modules can be imported into the benchmark.py script to be used in the benchmarking process.
As an example, we added some simple MIL methods (linear + mean, linear + max) below.

"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

def get_activation_function(activation_name: str):
    """Return the corresponding activation function from a string name."""
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class linear_evaluation_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, n_out),
            get_activation_function(activation_function)
        )
        self.head = nn.Linear(n_out, n_out)
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class linear_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class mean_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class max_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        pooled_embeddings = embeddings.max(dim=1)[0]
        scores = self.head(pooled_embeddings)
        return scores


class lse_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, r: float = 1.0, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        lse_pooling = self.r * torch.logsumexp(embeddings / self.r, dim=1)
        scores = self.head(lse_pooling)
        return scores


class lstm_mil(nn.Module):
    use_lens = True

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, lstm_dim: int = 128, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        lens_cpu = lens.cpu().to(dtype=torch.int64)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lens_cpu, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embeddings)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        pooled_embeddings = hidden[-1]
        scores = self.head(pooled_embeddings)
        return scores


class deepset_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
        embeddings = self.encoder(bags)
        phi_output = self.deepset_phi(embeddings)
        pooled_embeddings = phi_output.sum(dim=1)
        scores = self.deepset_rho(pooled_embeddings)
        return scores


class distributionpooling_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * 2),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * 2, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        embeddings = self.encoder(bags)
        mean_embeddings = embeddings.mean(dim=1)
        variance_embeddings = embeddings.var(dim=1)
        pooled_embeddings = torch.cat([mean_embeddings, variance_embeddings], dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class dsmil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, num_instances, _ = bags.size()
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        max_score, max_idx = instance_scores.max(dim=1)
        critical_instance = bags[torch.arange(batch_size), max_idx]
        instance_embeddings = instance_features.view(batch_size, num_instances, -1)
        critical_embeddings = instance_embeddings[torch.arange(batch_size), max_idx]
        attention_weights = F.softmax(self.attention(instance_embeddings - critical_embeddings.unsqueeze(1)), dim=1)
        bag_embeddings = (attention_weights * instance_embeddings).sum(dim=1)
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
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

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

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        embeddings = self.encoder(bags)
        attention_scores = self.attention(embeddings)
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores


class cluster_pooling_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, n_clusters: int = 4, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim * n_clusters),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * n_clusters, n_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        pooled_embeddings = []

        for i in range(batch_size):
            embeddings = self.encoder(bags[i])
            n_clusters = min(self.kmeans.n_clusters, n_patches)
            if n_clusters < 2:
                cluster_embedding = embeddings.mean(dim=0)
                pooled_embeddings.append(cluster_embedding)
                continue
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings.detach().cpu().numpy())
            cluster_embeddings = []
            for j in range(n_clusters):
                cluster_indices = torch.tensor(clusters == j, dtype=torch.bool, device=embeddings.device)
                if cluster_indices.sum() > 0:
                    cluster_embedding = embeddings[cluster_indices].mean(dim=0)
                else:
                    cluster_embedding = torch.zeros(embeddings.size(-1), device=embeddings.device)
                cluster_embeddings.append(cluster_embedding)
            pooled_embeddings.append(torch.cat(cluster_embeddings, dim=0))
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)
        scores = self.head(pooled_embeddings)
        return scores


class gated_attention_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU') -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        embeddings = self.encoder(bags)
        attention_V = self.attention_V(embeddings)
        attention_U = self.attention_U(embeddings)
        attention_weights = self.attention_weights(attention_V * attention_U).softmax(dim=1)
        pooled_embeddings = (embeddings * attention_weights).sum(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        embeddings = self.encoder(bags)
        attention_V = self.attention_V(embeddings)
        attention_U = self.attention_U(embeddings)
        attention_weights = self.attention_weights(attention_V * attention_U)
        if apply_softmax:
            attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights


class topk_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, k: int = 20, dropout_p: float = 0.1, activation_function='ReLU'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
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
        
        # Pass through the head
        scores = self.head(weighted_sum)
        return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores = F.softmax(scores, dim=1)
        return scores

class learnable_topk_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, initial_k: int = 20, dropout_p: float = 0.1, activation_function='ReLU'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
        self.attention = nn.Sequential(
            nn.Linear(z_dim, 1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        
        # Always learnable k
        self.k_param = nn.Parameter(torch.tensor(float(initial_k)))
        
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
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
        
        # Pass through the head
        scores = self.head(weighted_mean)
        return scores

    def calculate_attention(self, bags, lens=None, apply_softmax=None):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores = F.softmax(scores, dim=1)
        return scores
        
class hierarchical_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, region_size: int = 4, dropout_p: float = 0.1, activation_function='ReLU'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        n_regions = n_patches // self.region_size

        region_embeddings = []
        for i in range(n_regions):
            region_patches = embeddings[:, i * self.region_size: (i + 1) * self.region_size]
            attention_weights = F.softmax(self.region_attention(region_patches), dim=1)
            region_embedding = torch.sum(attention_weights * region_patches, dim=1)
            region_embeddings.append(self.region_head(region_embedding))

        region_embeddings = torch.stack(region_embeddings, dim=1)
        region_attention_weights = F.softmax(self.slide_attention(region_embeddings), dim=1)
        slide_embedding = torch.sum(region_attention_weights * region_embeddings, dim=1)
        scores = self.slide_head(slide_embedding)
        return scores


class hierarchical_cluster_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, n_clusters: int = 10, dropout_p: float = 0.1, activation_function='ReLU'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
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
            embeddings = self.encoder(bags[i])
            n_clusters = min(self.kmeans.n_clusters, n_patches)
            if n_clusters < 2:
                region_embeddings = embeddings.mean(dim=0, keepdim=True).unsqueeze(0)
                all_region_embeddings.append(region_embeddings)
                continue
            clusters = self.kmeans.fit_predict(embeddings.detach().cpu().numpy())
            region_embeddings = []

            for j in range(n_clusters):
                cluster_mask = torch.tensor(clusters == j, dtype=torch.bool, device=embeddings.device)
                cluster_patches = embeddings[cluster_mask]

                if cluster_patches.size(0) == 0:
                    continue

                attention_weights = F.softmax(self.region_attention(cluster_patches), dim=0)
                region_embedding = torch.sum(attention_weights * cluster_patches, dim=0)

                if region_embedding.size(0) == 1:
                    region_embeddings.append(region_embedding.unsqueeze(0))
                else:
                    region_embeddings.append(self.region_head(region_embedding.unsqueeze(0)))

            if region_embeddings:
                all_region_embeddings.append(torch.cat(region_embeddings, dim=0))

        if len(all_region_embeddings) > 0:
            region_embeddings = torch.stack(all_region_embeddings, dim=0)
        else:
            raise ValueError("No regions were found. Check your clustering or input data.")

        if region_embeddings.dim() == 2:
            region_embeddings = region_embeddings.unsqueeze(1)

        region_attention_weights = F.softmax(self.slide_attention(region_embeddings), dim=1)
        slide_embedding = torch.sum(region_attention_weights * region_embeddings, dim=1)
        scores = self.slide_head(slide_embedding)
        return scores


class weighted_mean_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU'):
        super(weighted_mean_mil, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function)
        )
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
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 512, dropout_p: float = 0.25, activation_function='ReLU'):
        super(clam_mil, self).__init__()
        self.n_out = n_out
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            get_activation_function(activation_function),
            nn.Dropout(dropout_p)
        )
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
        embeddings = self.encoder(bags)
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
        embeddings = self.encoder(bags)
        cluster_predictions = []
        for i in range(self.n_out):
            cluster_pred = self.instance_cluster[i](embeddings)
            cluster_predictions.append(cluster_pred)
        return cluster_predictions

    def calculate_attention(self, bags, lens=None, apply_softmax=False):
        embeddings = self.encoder(bags)
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
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 512, dropout_p: float = 0.25, n_branches: int = 3, activation_function='ReLU'):
        super(clam_mil_mb, self).__init__()
        self.n_out = n_out
        self.n_branches = n_branches
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_feats, z_dim),
                get_activation_function(activation_function),
                nn.Dropout(dropout_p)
            ) for _ in range(n_branches)
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


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)