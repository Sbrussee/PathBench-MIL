##MIL-friendly augmentations that can be used during training

def patch_dropout(features, dropout_rate=0.5):
    mask = np.random.binomial(1, 1 - dropout_rate, size=features.shape[0])
    return features[mask.astype(bool)]

def shuffle_patches(features):
    np.random.shuffle(features)
    return features

def add_gaussian_noise(features, std_dev=0.01):
    noise = np.random.normal(0, std_dev, features.shape)
    return features + noise

def random_scaling(features, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return features * scale

def feature_masking(features, mask_rate=0.2):
    mask = np.random.binomial(1, mask_rate, size=features.shape)
    return features * (1 - mask)

def feature_dropout(features, dropout_rate=0.5):
    if np.random.rand() < dropout_rate:
        return np.zeros_like(features)
    return features