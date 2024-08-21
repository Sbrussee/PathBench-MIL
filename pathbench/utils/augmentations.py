import torch
import numpy as np

def patch_dropout(bag: torch.Tensor, dropout_rate: float = 0.5) -> torch.Tensor:
    """
    Randomly drops instances from the bag based on the dropout_rate.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    dropout_rate (float): The probability of dropping each instance in the bag.

    Returns:
    torch.Tensor: A new bag with a subset of the original instances, depending on the dropout rate.
    """
    mask = torch.from_numpy(np.random.binomial(1, 1 - dropout_rate, size=bag.shape[0])).bool()
    return bag[mask]

def add_gaussian_noise(bag: torch.Tensor, std_dev: float = 0.01) -> torch.Tensor:
    """
    Adds Gaussian noise to each feature in the bag.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    std_dev (float): The standard deviation of the Gaussian noise to be added.

    Returns:
    torch.Tensor: A new bag with Gaussian noise added to each feature.
    """
    noise = torch.randn_like(bag) * std_dev
    return bag + noise

def random_scaling(bag: torch.Tensor, scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:
    """
    Scales all features in the bag by a random factor within a given range.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    scale_range (tuple): A tuple of two floats (min_scale, max_scale) representing the range within which the scaling factor is chosen.

    Returns:
    torch.Tensor: A new bag with all features scaled by the chosen factor.
    """
    scale = torch.FloatTensor(1).uniform_(*scale_range).item()
    return bag * scale

def feature_masking(bag: torch.Tensor, mask_rate: float = 0.2) -> torch.Tensor:
    """
    Randomly masks features (sets them to zero) in the bag based on a given rate.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    mask_rate (float): The probability of masking each feature in the bag.

    Returns:
    torch.Tensor: A new bag with some features masked (set to zero).
    """
    mask = torch.from_numpy(np.random.binomial(1, 1 - mask_rate, size=bag.shape)).float()
    return bag * mask

def feature_dropout(bag: torch.Tensor, dropout_rate: float = 0.5) -> torch.Tensor:
    """
    With a certain probability, sets all features of the entire bag to zero.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    dropout_rate (float): The probability of setting all features in the bag to zero.

    Returns:
    torch.Tensor: Either the original bag or a bag of zeros, depending on the dropout rate.
    """
    if torch.rand(1).item() < dropout_rate:
        return torch.zeros_like(bag)
    return bag

def patchwise_scaling(bag: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """
    Scales each instance in the bag by a different random factor within a given range.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    scale_range (tuple): A tuple of two floats (min_scale, max_scale) representing the range within which the scaling factor is chosen for each instance.

    Returns:
    torch.Tensor: A new bag with each instance scaled by its own randomly chosen factor.
    """
    scales = torch.FloatTensor(bag.size(0), 1).uniform_(*scale_range)
    return bag * scales

def feature_permutation(bag: torch.Tensor) -> torch.Tensor:
    """
    Permutes the order of instances within the bag.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.

    Returns:
    torch.Tensor: A new bag with the instances permuted (shuffled).
    """
    return bag[torch.randperm(bag.size(0))]


def patch_mixing(bag: torch.Tensor, mixing_rate: float = 0.5) -> torch.Tensor:
    """
    Randomly selects and mixes two instances within the same bag based on a given mixing rate.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    mixing_rate (float): The probability of selecting features from one instance over another during mixing.

    Returns:
    torch.Tensor: A new bag where one instance is replaced by a mix of two randomly selected instances.
    """
    indices = torch.randperm(bag.size(0))[:2]
    instance1, instance2 = bag[indices]
    mask = torch.from_numpy(np.random.binomial(1, mixing_rate, size=instance1.shape)).bool()
    mixed_instance = torch.where(mask, instance1, instance2)
    bag[indices[0]] = mixed_instance
    return bag

def cutmix(bag: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Concatenates portions of two randomly selected instances within the same bag based on a random lambda.

    Parameters:
    bag (torch.Tensor): A 2D tensor of shape (num_instances, num_features) representing a bag of instances.
    alpha (float): A value controlling the range of the lambda parameter, which determines how much of each instance to mix.

    Returns:
    torch.Tensor: A new bag where one instance is replaced by a combination of two randomly selected instances.
    """
    if bag.size(0) < 2:
        return bag  # Not enough instances to perform cutmix
    indices = torch.randperm(bag.size(0))[:2]
    instance1, instance2 = bag[indices]
    lam = torch.FloatTensor(1).uniform_(alpha, 1 - alpha).item()
    cut_point = int(instance1.size(0) * lam)
    mixed_instance = torch.cat([instance1[:cut_point], instance2[cut_point:]], dim=0)
    bag[indices[0]] = mixed_instance
    return bag