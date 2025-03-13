import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.mixture import GaussianMixture


# =============================================================================
# Helper Functions
# =============================================================================

def get_activation_function(activation_name: str):
    """
    Retrieve a torch activation function by name.

    Args:
        activation_name (str): The name of the activation function (e.g., 'ReLU').

    Returns:
        nn.Module or None: The activation function module instance if provided; otherwise, None.

    Raises:
        ValueError: If the activation function name is not supported.
    """
    if activation_name is None:
        return None
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Unsupported activation function: {activation_name}")


def build_encoder(n_feats: int,
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
      - Optional Batch Normalization (if use_batchnorm is True)
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
    encoder_layers = int(encoder_layers)
    layers = []
    for i in range(int(encoder_layers)):
        in_features = n_feats if i == 0 else z_dim
        layers.append(nn.Linear(in_features, z_dim))
        if activation_function is not None:
            layers.append(get_activation_function(activation_function))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(z_dim))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


def initialize_weights(module: nn.Module, activation_function: str):
    """
    Initialize the weights for the given module (or submodules).

    For Linear layers:
      - If the activation function includes 'lu' (e.g., Linear Unit), use Kaiming (He) initialization.
      - Otherwise, use Xavier (Glorot) initialization.
    For BatchNorm layers:
      - Initialize weights to 1 and biases to 0.

    Args:
        module (nn.Module): The module (or submodule) whose weights are to be initialized.
        activation_function (str): The activation function used in the model (to decide on initialization).
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if activation_function and 'lu' in activation_function.lower():
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# =============================================================================
# MIL Aggregator Models
# =============================================================================

class linear_mil(nn.Module):
    """
    linear_mil (Linear Evaluation MIL Model)

    This model consists of:
      1) An encoder (a stack of linear layers with optional activation, batch norm, and dropout)
      2) A linear head that produces final predictions from the mean-pooled embedding.

    The architecture adapts based on the 'goal':
      - 'classification': Uses batch norm in the encoder and outputs n_out classes.
      - 'survival' or 'survival_discrete': Outputs n_out scores.
      - 'regression': Outputs a single value.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = None,
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the linear_mil model.

        Args:
            n_feats (int): Number of features per instance.
            n_out (int): Number of output predictions.
            z_dim (int): Latent dimension of the encoder.
            dropout_p (float): Dropout probability.
            activation_function (str): Name of the activation function to use.
            encoder_layers (int): Number of layers in the encoder.
            goal (str): The task goal ('classification', 'survival', 'survival_discrete', or 'regression').
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Linear(z_dim, n_out)
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using the helper function."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear_mil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats], where B is the batch size
                                 and N is the number of instances/patches.

        Returns:
            torch.Tensor: The output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class mean_mil(nn.Module):
    """
    mean_mil (Mean-Pooling MIL Model)

    This model encodes each instance using an MLP and aggregates the embeddings by taking their mean.
    A final head then produces the output.

    Suitable for:
      - 'classification': Includes batch norm in the head.
      - 'survival' / 'regression' / 'survival_discrete': The final layer is adapted accordingly.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the mean_mil model.

        Args:
            n_feats (int): Number of features per instance.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function to use.
            encoder_layers (int): Number of layers in the encoder.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mean_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores


class max_mil(nn.Module):
    """
    max_mil (Max-Pooling MIL Model)

    This model encodes each instance and aggregates the embeddings by taking the maximum
    value across instances. The final head then produces the output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the max_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function to use.
            encoder_layers (int): Number of layers in the encoder.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for max_mil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        pooled_embeddings = embeddings.max(dim=1)[0]
        scores = self.head(pooled_embeddings)
        return scores


class lse_mil(nn.Module):
    """
    lse_mil (Log-Sum-Exp MIL Model)

    This model uses Log-Sum-Exp pooling—a smooth approximation to max-pooling—to aggregate
    instance embeddings. The pooled embedding is then passed through a final head to produce the output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 r: float = 1.0,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the lse_mil model.

        Args:
            n_feats (int): Number of features per instance.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            r (float): Temperature parameter for LSE pooling.
            activation_function (str): Activation function to use.
            encoder_layers (int): Number of layers in the encoder.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.r = r
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for lse_mil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        lse_pooling = self.r * torch.logsumexp(embeddings / self.r, dim=1)
        scores = self.head(lse_pooling)
        return scores


class lstm_mil(nn.Module):
    """
    lstm_mil (LSTM-based MIL Model)

    This model employs an LSTM to process the sequence of instance embeddings,
    handling variable-length bags. The last hidden state of the LSTM is used for the bag-level prediction.
    """
    use_lens = True

    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 lstm_dim: int = 128,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the lstm_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent dimension of the encoder.
            lstm_dim (int): Hidden dimension for the LSTM.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.lstm = nn.LSTM(z_dim, lstm_dim, batch_first=True)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(lstm_dim),
                nn.Dropout(dropout_p),
                nn.Linear(lstm_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(lstm_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(lstm_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for lstm_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for lstm_mil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats].
            lens (torch.Tensor): Lengths of each bag (for packing), shape [B].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lens.cpu().to(dtype=torch.int64), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.lstm(packed_embeddings)
        pooled_embeddings = hidden[-1]
        scores = self.head(pooled_embeddings)
        return scores


class deepset_mil(nn.Module):
    """
    deepset_mil (Deep Sets MIL Model)

    This model follows the Deep Sets paradigm:
      1) Applies a per-instance network (phi) to compute embeddings.
      2) Aggregates (sums) the embeddings over all instances.
      3) Uses a final network (rho) to produce bag-level predictions.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the deepset_mil model.

        Args:
            n_feats (int): Number of features per instance.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.deepset_phi = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            get_activation_function(activation_function),
            nn.Linear(z_dim, z_dim)
        )
        if goal == 'classification':
            self.deepset_rho = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                get_activation_function(activation_function),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.deepset_rho = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.deepset_rho = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for deepset_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for deepset_mil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        phi_output = self.deepset_phi(embeddings)
        pooled_embeddings = phi_output.sum(dim=1)
        scores = self.deepset_rho(pooled_embeddings)
        return scores


class distributionpooling_mil(nn.Module):
    """
    distributionpooling_mil (Distribution Pooling MIL Model)

    This model aggregates instance embeddings by computing both the mean and variance
    across instances. These statistics are concatenated and passed through a final head.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the distributionpooling_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim * 2),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim * 2, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim * 2, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim * 2, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for distributionpooling_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributionpooling_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
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
    dsmil (Dual-Stream MIL Model)

    This model processes instance embeddings in two streams:
      1) An instance-level stream that scores each instance.
      2) A bag-level stream that uses attention (computed relative to a critical instance)
         to obtain a bag representation.
      
    The final prediction is computed from a 1D convolution over the bag embedding.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the dsmil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.instance_encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                              activation_function, dropout_p, use_bn)
        self.instance_classifier = nn.Linear(z_dim, 1)
        self.attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        self.conv1d = nn.Conv1d(z_dim, z_dim, kernel_size=1)
        if goal == 'classification':
            self.bag_classifier = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.bag_classifier = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.bag_classifier = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for dsmil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for dsmil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            return_attention (bool): If True, also return attention weights.

        Returns:
            torch.Tensor or tuple: Output scores, and optionally attention weights.
        """
        batch_size, num_instances, _ = bags.size()
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_features = instance_features.view(batch_size, num_instances, -1)
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        max_score, max_idx = instance_scores.max(dim=1)
        critical_embeddings = instance_features[torch.arange(batch_size), max_idx]
        attention_weights = F.softmax(
            self.attention(instance_features - critical_embeddings.unsqueeze(1)),
            dim=1
        )
        bag_embeddings = (attention_weights * instance_features).sum(dim=1)
        bag_embeddings = bag_embeddings.unsqueeze(-1)
        conv_out = self.conv1d(bag_embeddings).squeeze(-1)
        scores = self.bag_classifier(conv_out)
        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Compute attention weights for dsmil without classification.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused (for compatibility).
            apply_softmax (bool): Whether to apply softmax to the attention scores.

        Returns:
            torch.Tensor: Attention scores.
        """
        batch_size, num_instances, _ = bags.size()
        instance_features = self.instance_encoder(bags.view(-1, bags.size(-1)))
        instance_features = instance_features.view(batch_size, num_instances, -1)
        instance_scores = self.instance_classifier(instance_features).view(batch_size, num_instances)
        _, max_idx = instance_scores.max(dim=1)
        critical_embeddings = instance_features[torch.arange(batch_size), max_idx]
        attention_scores = self.attention(instance_features - critical_embeddings.unsqueeze(1))
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores


class varmil(nn.Module):
    """
    varmil (Variance-based Attention MIL Model)

    This model computes a weighted mean of instance embeddings using attention,
    then also computes the weighted variance around that mean. The mean and variance
    are concatenated and passed through a final head to produce the output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the varmil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.attention = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, 1)
        )
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(z_dim * 2),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim * 2, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim * 2, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim * 2, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for varmil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for varmil.

        Args:
            bags (torch.Tensor): Tensor of shape [B, N, n_feats].
            return_attention (bool): If True, also return attention weights.

        Returns:
            torch.Tensor or tuple: Output scores, and optionally attention weights.
        """
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

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Compute attention scores for varmil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused (for compatibility).
            apply_softmax (bool): Whether to apply softmax.

        Returns:
            torch.Tensor: Attention scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_scores = self.attention(embeddings)
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores


class perceiver_mil(nn.Module):
    """
    perceiver_mil (Perceiver MIL Model)

    This model uses learnable latent vectors and cross-attention to aggregate instance features.
    The latent representations are then processed by a Transformer, and the output is produced
    from the CLS token.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 latent_dim: int = 128,
                 num_latents: int = 16,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the perceiver_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent dimension used in the encoder.
            latent_dim (int): Dimension of the learnable latent vectors.
            num_latents (int): Number of latent vectors.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super(perceiver_mil, self).__init__()
        use_bn = (goal == 'classification')
        self.cls_token = nn.Parameter(torch.randn(1, latent_dim))
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, 
                                                     num_heads=num_heads, 
                                                     dropout=dropout_p)
        self.transformer_layers = TransformerEncoder(
            TransformerEncoderLayer(d_model=latent_dim, 
                                    nhead=num_heads, 
                                    dim_feedforward=z_dim, 
                                    dropout=dropout_p),
            num_layers=num_layers
        )
        self.input_projection = nn.Linear(n_feats, latent_dim)
        if goal == 'classification':
            self.output_projection = nn.Linear(latent_dim, n_out)
        elif goal in ['survival', 'regression']:
            self.output_projection = nn.Linear(latent_dim, 1)
        elif goal == 'survival_discrete':
            self.output_projection = nn.Linear(latent_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for perceiver_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for perceiver_mil.

        Args:
            bags (torch.Tensor): Input tensor of shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        x = self.input_projection(bags.view(-1, n_feats)).view(batch_size, n_patches, -1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        latents = torch.cat([cls_tokens, self.latents.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
        latents = latents.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        latents, _ = self.cross_attention(latents, x, x)
        latents = self.transformer_layers(latents)
        cls_output = latents[0]
        output = self.output_projection(cls_output)
        return output


class gated_attention_mil(nn.Module):
    """
    gated_attention_mil (Gated Attention MIL Model)

    This model uses two parallel attention networks (one using Tanh and one using Sigmoid)
    to compute a gating mechanism. The resulting attention weights are used to pool instance embeddings,
    and the final head produces the output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the gated_attention_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.attention_V = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(z_dim, 1)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for gated_attention_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated_attention_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        A_V = self.attention_V(embeddings)
        A_U = self.attention_U(embeddings)
        att = self.attention_weights(A_V * A_U).softmax(dim=1)
        pooled_embeddings = (embeddings * att).sum(dim=1)
        scores = self.head(pooled_embeddings)
        return scores

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Calculate attention weights for gated_attention_mil without final classification.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused.
            apply_softmax (bool): If True, apply softmax to the raw attention scores.

        Returns:
            torch.Tensor: Attention weights.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        A_V = self.attention_V(embeddings)
        A_U = self.attention_U(embeddings)
        att = self.attention_weights(A_V * A_U)
        if apply_softmax:
            att = F.softmax(att, dim=1)
        return att


class topk_mil(nn.Module):
    """
    topk_mil (Top-k MIL Model)

    This model scores each instance, selects the top-k instances based on their scores,
    applies a softmax on the top-k scores, and computes a weighted sum of their embeddings.
    The weighted sum is passed through a final head to produce the output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 k: int = 20,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the topk_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            k (int): Number of top instances to consider.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.attention = nn.Linear(z_dim, 1)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.k = k
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for topk_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for topk_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            return_attention (bool): If True, also return the top-k attention weights.

        Returns:
            torch.Tensor or tuple: Output scores, and optionally attention weights.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores_ = self.attention(embeddings).squeeze(-1)
        k = min(self.k, n_patches)
        topk_scores, topk_indices = torch.topk(scores_, k, dim=1)
        topk_embeddings = torch.gather(
            embeddings, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
        )
        topk_attention_weights = F.softmax(topk_scores, dim=1)
        weighted_sum = torch.sum(topk_embeddings * topk_attention_weights.unsqueeze(-1), dim=1)
        out = self.head(weighted_sum)
        if return_attention:
            return out, topk_attention_weights
        else:
            return out

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Calculate raw attention scores for topk_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused.
            apply_softmax (bool): If True, apply softmax.

        Returns:
            torch.Tensor: Attention scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores_ = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores_ = F.softmax(scores_, dim=1)
        return scores_


class air_mil(nn.Module):
    """
    air_mil (Adaptive Instance Ranking MIL Model)

    This model introduces a learnable parameter 'k' which determines how many top instances
    to consider. The aggregation follows a similar approach to topk_mil.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 initial_k: int = 20,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the air_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            initial_k (int): Initial value for k (learnable).
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.attention = nn.Linear(z_dim, 1)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.k_param = nn.Parameter(torch.tensor(float(initial_k)), requires_grad=True)
        self.goal = goal
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for air_mil."""
        initialize_weights(self, self.goal)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for air_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            return_attention (bool): If True, also return the attention weights.

        Returns:
            torch.Tensor or tuple: Output scores, and optionally attention weights.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores_ = self.attention(embeddings).squeeze(-1)
        k_ = torch.clamp(self.k_param, 1, n_patches).int()
        topk_scores, topk_indices = torch.topk(scores_, k_, dim=1)
        topk_embeddings = torch.gather(
            embeddings, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
        )
        topk_attention_weights = F.softmax(topk_scores, dim=1)
        weighted_sum = torch.sum(topk_embeddings * topk_attention_weights.unsqueeze(-1), dim=1)
        weighted_mean = weighted_sum / torch.sum(topk_attention_weights, dim=1, keepdim=True)
        out = self.head(weighted_mean)
        if return_attention:
            return out, topk_attention_weights
        else:
            return out

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Calculate attention scores for air_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused.
            apply_softmax (bool): If True, apply softmax.

        Returns:
            torch.Tensor: Attention scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        scores_ = self.attention(embeddings).squeeze(-1)
        if apply_softmax:
            scores_ = F.softmax(scores_, dim=1)
        return scores_


class il_mil(nn.Module):
    """
    il_mil (Instance-Level MIL Model)

    This model classifies each instance separately using the encoder and a classifier,
    then averages the instance-level predictions to produce a bag-level output.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the il_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super().__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.instance_classifier = nn.Linear(z_dim, n_out)
        elif goal in ['survival', 'regression']:
            self.instance_classifier = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.instance_classifier = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self.activation_function = activation_function
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for il_mil."""
        initialize_weights(self, self.activation_function)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for il_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Bag-level output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        instance_scores = self.instance_classifier(embeddings)
        bag_scores = instance_scores.mean(dim=1)
        return bag_scores


class weighted_mean_mil(nn.Module):
    """
    weighted_mean_mil (Weighted Mean MIL Model)

    This model computes the variance of the instance embeddings and uses the inverse variance
    as a weight for each instance. A weighted mean is then computed and passed through a final head.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the weighted_mean_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super(weighted_mean_mil, self).__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self.activation_function = activation_function
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for weighted_mean_mil."""
        initialize_weights(self, self.activation_function)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for weighted_mean_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
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
    clam_mil (CLAM-MIL Model)

    This model implements a clustering-constrained attention MIL approach in which
    each output class has its own attention branch and classifier. Instance embeddings are
    aggregated via a weighted sum computed separately for each class.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the clam_mil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs (classes).
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super(clam_mil, self).__init__()
        self.n_out = n_out
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.attention_U = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh()
        )
        self.attention_branches = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        self.classifiers = nn.ModuleList()
        self.instance_cluster = nn.ModuleList()
        for _ in range(n_out):
            if goal == 'classification':
                self.classifiers.append(nn.Linear(z_dim, 1))
            elif goal in ['survival', 'regression']:
                self.classifiers.append(nn.Linear(z_dim, 1))
            elif goal == 'survival_discrete':
                self.classifiers.append(nn.Linear(z_dim, 1))
            else:
                raise ValueError(f"Unsupported goal: {goal}")
        for _ in range(n_out):
            self.instance_cluster.append(nn.Linear(z_dim, 2))
        self.goal = goal
        self.activation_function = activation_function
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for clam_mil."""
        initialize_weights(self, self.activation_function)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for clam_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            return_attention (bool): If True, return average attention weights.

        Returns:
            torch.Tensor or tuple: Output scores, and optionally attention weights.
        """
        batch_size, n_patches, _ = bags.shape
        embeddings = self.encoder(bags.view(-1, bags.size(-1)))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        slide_level_reps = []
        attention_weights_list = []
        for i in range(self.n_out):
            A_U = self.attention_U(embeddings)
            A_V = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i]((A_U * A_V)).softmax(dim=1)
            attention_weights_list.append(attention_scores)
            slide_rep = torch.sum(attention_scores * embeddings, dim=1)
            slide_level_reps.append(slide_rep)
        slide_level_reps = torch.stack(slide_level_reps, dim=1)
        all_scores = []
        for i in range(self.n_out):
            sc = self.classifiers[i](slide_level_reps[:, i, :])
            all_scores.append(sc)
        scores = torch.cat(all_scores, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=0).mean(dim=0)
        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def cluster_patches(self, bags: torch.Tensor):
        """
        Predict instance-level cluster assignments for interpretability.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            list: A list of cluster predictions per class.
        """
        batch_size, n_patches, _ = bags.shape
        embeddings = self.encoder(bags.view(-1, bags.size(-1)))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        cluster_predictions = []
        for i in range(self.n_out):
            cluster_pred = self.instance_cluster[i](embeddings)
            cluster_predictions.append(cluster_pred)
        return cluster_predictions

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Calculate attention weights for clam_mil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused.
            apply_softmax (bool): If True, apply softmax to attention scores.

        Returns:
            torch.Tensor: Average attention weights across classes.
        """
        batch_size, n_patches, _ = bags.shape
        embeddings = self.encoder(bags.view(-1, bags.size(-1)))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        A_weights_list = []
        for i in range(self.n_out):
            A_U = self.attention_U(embeddings)
            A_V = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i](A_U * A_V)
            if apply_softmax:
                attention_scores = F.softmax(attention_scores, dim=1)
            A_weights_list.append(attention_scores)
        attention_weights = torch.mean(torch.stack(A_weights_list), dim=0)
        return attention_weights


class clam_mil_mb(nn.Module):
    """
    clam_mil_mb (CLAM-MIL with Multiple Branches Model)

    This variant replicates the CLAM logic across several branches and then averages the results.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 dropout_p: float = 0.1,
                 n_branches: int = 3,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the clam_mil_mb model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability.
            n_branches (int): Number of branches.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super(clam_mil_mb, self).__init__()
        self.n_out = n_out
        self.n_branches = n_branches
        use_bn = (goal == 'classification')
        self.encoders = nn.ModuleList([
            build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, use_bn)
            for _ in range(n_branches)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(z_dim, z_dim), nn.Sigmoid()) for _ in range(n_branches)
        ])
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(z_dim, z_dim), nn.Tanh()) for _ in range(n_branches)
        ])
        self.attention_branches = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)]) for _ in range(n_branches)
        ])
        self.classifiers = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)]) for _ in range(n_branches)
        ])
        self.instance_cluster = nn.ModuleList([
            nn.ModuleList([nn.Linear(z_dim, 2) for _ in range(n_out)]) for _ in range(n_branches)
        ])
        self.goal = goal
        self.activation_function = activation_function
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for clam_mil_mb."""
        initialize_weights(self, self.activation_function)

    def forward(self, bags: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for clam_mil_mb.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            return_attention (bool): If True, return attention weights.

        Returns:
            torch.Tensor or tuple: Combined output scores, and optionally attention weights.
        """
        batch_size, n_patches, _ = bags.shape
        all_scores = []
        all_attention_weights = []
        for branch_idx in range(self.n_branches):
            embeddings = self.encoders[branch_idx](bags.view(-1, bags.size(-1)))
            embeddings = embeddings.view(batch_size, n_patches, -1)
            slide_level_reps = []
            attention_weights_list = []
            for i in range(self.n_out):
                A_U = self.attention_U[branch_idx](embeddings)
                A_V = self.attention_V[branch_idx](embeddings)
                att_scores = self.attention_branches[branch_idx][i]((A_U * A_V)).softmax(dim=1)
                attention_weights_list.append(att_scores)
                sl_rep = torch.sum(att_scores * embeddings, dim=1)
                slide_level_reps.append(sl_rep)
            slide_level_reps = torch.stack(slide_level_reps, dim=1)
            scores_ = []
            for i in range(self.n_out):
                sc = self.classifiers[branch_idx][i](slide_level_reps[:, i, :])
                scores_.append(sc)
            scores_ = torch.cat(scores_, dim=1)
            attention_weights_ = torch.cat(attention_weights_list, dim=1)
            all_scores.append(scores_)
            all_attention_weights.append(attention_weights_)
        combined_scores = torch.mean(torch.stack(all_scores), dim=0)
        combined_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
        if return_attention:
            return combined_scores, combined_attention_weights
        else:
            return combined_scores

    def calculate_attention(self, bags: torch.Tensor, lens=None, apply_softmax: bool = False):
        """
        Calculate attention weights for clam_mil_mb.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].
            lens: Unused.
            apply_softmax (bool): If True, apply softmax.

        Returns:
            torch.Tensor: Combined attention weights.
        """
        batch_size, n_patches, _ = bags.shape
        all_attention_weights = []
        for branch_idx in range(self.n_branches):
            embeddings = self.encoders[branch_idx](bags.view(-1, bags.size(-1)))
            embeddings = embeddings.view(batch_size, n_patches, -1)
            attention_weights_list = []
            for i in range(self.n_out):
                A_U = self.attention_U[branch_idx](embeddings)
                A_V = self.attention_V[branch_idx](embeddings)
                scores_ = self.attention_branches[branch_idx][i](A_U * A_V)
                if apply_softmax:
                    scores_ = F.softmax(scores_, dim=1)
                attention_weights_list.append(scores_)
            attention_weights_ = torch.cat(attention_weights_list, dim=1)
            all_attention_weights.append(attention_weights_)
        combined_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
        return combined_attention_weights


class transmil(nn.Module):
    """
    transmil (Transformer-based MIL Model)

    This model employs an encoder followed by a Transformer to capture contextual relationships
    among instance embeddings. The mean-pooled representation from the Transformer is used for the final prediction.
    """
    def __init__(self,
                 n_feats: int,
                 n_out: int,
                 z_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout_p: float = 0.1,
                 activation_function: str = 'ReLU',
                 encoder_layers: int = 1,
                 goal: str = 'classification'):
        """
        Initialize the transmil model.

        Args:
            n_feats (int): Input feature dimension.
            n_out (int): Number of outputs.
            z_dim (int): Latent dimension for the encoder.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout_p (float): Dropout probability.
            activation_function (str): Activation function.
            encoder_layers (int): Number of encoder layers.
            goal (str): The task goal.
        """
        super(transmil, self).__init__()
        use_bn = (goal == 'classification')
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers,
                                     activation_function, dropout_p, use_bn)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=z_dim, 
                                    nhead=num_heads, 
                                    dim_feedforward=z_dim, 
                                    dropout=dropout_p),
            num_layers=num_layers
        )
        if goal == 'classification':
            self.head = nn.Sequential(
                nn.BatchNorm1d(z_dim),
                nn.Dropout(dropout_p),
                nn.Linear(z_dim, n_out)
            )
        elif goal in ['survival', 'regression']:
            self.head = nn.Linear(z_dim, 1)
        elif goal == 'survival_discrete':
            self.head = nn.Linear(z_dim, n_out)
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        self.goal = goal
        self.activation_function = activation_function
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for transmil."""
        initialize_weights(self, self.activation_function)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transmil.

        Args:
            bags (torch.Tensor): Input tensor with shape [B, N, n_feats].

        Returns:
            torch.Tensor: Output scores.
        """
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.transformer(embeddings)
        embeddings = embeddings.permute(1, 0, 2)
        pooled_embeddings = embeddings.mean(dim=1)
        scores = self.head(pooled_embeddings)
        return scores
