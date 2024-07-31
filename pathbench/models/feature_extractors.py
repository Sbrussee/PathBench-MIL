from slideflow import mil
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors import register_torch
from torchvision.models.resnet import Bottleneck, ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, ViTModel, AutoModel
from huggingface_hub import login, hf_hub_download
import os
import yaml
import math
import functools
from functools import reduce
from operator import mul

keys = yaml.load(open("keys.yaml"), Loader=yaml.FullLoader)

# Directory to save pretrained weights
WEIGHTS_DIR = "pretrained_weights"
os.environ['TORCH_HOME'] = WEIGHTS_DIR
os.environ['HF_HOME'] = WEIGHTS_DIR


def get_pretrained_url_vit(key : str):
    """
    Get the URL for the pretrained weights of the Vision Transformer model

    Args:
        key (str): The key for the model
    
    Returns:
        str: The URL for the pretrained weights of the model
    """
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def vit_small(pretrained : bool, progress : bool, key : str, **kwargs):
    """
    Load the Vision Transformer model with small configuration

    Args:
        pretrained (bool): Whether to load the pretrained weights
        progress (bool): Whether to show the download progress
        key (str): The key for the model
        **kwargs: Additional keyword arguments
    
    Returns:
        VisionTransformer: The ViT-Small model
    """
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size = 224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )

    if pretrained:
        pretrained_url = get_pretrained_url_vit(key)
        weights_path = os.path.join(WEIGHTS_DIR, f"{key}.torch")
        if not os.path.exists(weights_path):
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            print(f"Downloading pretrained weights for {key}...")
            torch.hub.download_url_to_file(pretrained_url, weights_path)
            print("Pretrained weights downloaded successfully.")
        verbose = model.load_state_dict(torch.load(weights_path))
        print(verbose)
    return model


class ResNetTrunk(ResNet):
    """
    ResNet trunk without the final fully connected layer

    Parameters
    ----------
    *args : list
        Variable length argument list
    **kwargs : dict
        Keyword arguments

    Attributes
    ----------
    fc : nn.Linear
        Fully connected
    
    Methods
    -------
    forward(x)
        Forward pass of the model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to (batch_size, num_features)
        return x

def get_pretrained_url(key : str):
    """
    Get the URL for the pretrained weights of the model

    Args:
        key (str): The key for the model
    
    Returns:
        str: The URL for the pretrained weights of the model
    """
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def download_pretrained_weights(key : str, destination : str):
    """
    Download the pretrained weights for the model

    Args:
        key (str): The key for the model
        destination (str): The path to save the pretrained weights
    """
    pretrained_url = get_pretrained_url(key)
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(f"Downloading pretrained weights for {key}...")
        torch.hub.download_url_to_file(pretrained_url, destination)
        print("Pretrained weights downloaded successfully.")
    else:
        print(f"Pretrained weights for {key} already exist, skipping download.")

def resnet50(pretrained : bool, progress : bool, key : str, **kwargs):
    """
    Load the ResNet-50 model

    Args:
        pretrained (bool): Whether to load the pretrained weights
        progress (bool): Whether to show the download progress
        key (str): The key for the model
        **kwargs: Additional keyword arguments
    
    Returns:
        ResNetTrunk: The ResNet-50 model
    """
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_path = os.path.join(WEIGHTS_DIR, f"{key}.torch")
        download_pretrained_weights(key, weights_path)
        model.load_state_dict(torch.load(weights_path))
    return model

@register_torch
class dino(TorchFeatureExtractor):
    """
    Lunit-IO DINO feature extractor, with ViT-Small backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = "dino"

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = vit_small(pretrained=True, progress=False, key="DINO_p16")
        self.model.to('cuda')
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'dino',
            'kwargs': {}
        }


@register_torch
class barlow_twins(TorchFeatureExtractor):
    """
    Lunit-IO Barlow Twins feature extractor, with ResNet-50 backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : ResNet50 truncated
        The ResNet-50 model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'barlow_twins'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="BT")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose(
            [
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'barlow_twins',
            'kwargs': {}
        }


@register_torch
class mocov2(TorchFeatureExtractor):
    """
    Lunit-IO MoCoV2 feature extractor, with ResNet-50 backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : ResNet50 truncated
        The ResNet-50 model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'mocov2'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="MoCoV2")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose(
            [
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'mocov2',
            'kwargs': {}
        }


@register_torch
class swav(TorchFeatureExtractor):
    """
    Lunit-IO SwAV feature extractor, with ResNet-50 backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : ResNet50 truncated
        The ResNet-50 model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'swav'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="SwAV")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose(
            [
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'swav',
            'kwargs': {}
        }
    

@register_torch
class uni(TorchFeatureExtractor):
    """
    UNI feature extractor, with ViT-Large backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = "uni"

    def __init__(self, tile_px=256):
        super().__init__()

        login(token=keys['uni'])

        local_dir = WEIGHTS_DIR
        model_name = "uni.bin"
        model_temp_name = "pytorch_model.bin"
        model_path = os.path.join(local_dir, model_name)

        if not os.path.exists(model_path):
            temp_model_path = hf_hub_download(repo_id="MahmoodLab/UNI", filename=model_temp_name, local_dir=local_dir)
            os.rename(temp_model_path, model_path)
        
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, init_values=1e-5, num_classes=0
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.model.to('cuda')
        self.num_features = 1024 
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'uni',
            'kwargs': {}
        }

@register_torch
class phikon(TorchFeatureExtractor):
    """
    Phikon feature extractor, with ViT-Large backbone

    Parameters    
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'phikon'

    def __init__(self, tile_px=256):
        super().__init__()

        local_dir = WEIGHTS_DIR

        # Load the pre-trained phikon model
        self.base_model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.num_features = 768

        # Internal class for the modified model
        class PhikonEmbedder(nn.Module):
            def __init__(self, base_model, num_features):
                super(PhikonEmbedder, self).__init__()
                self.base_model = base_model
                self.num_features = num_features

            def forward(self, x):
                # Get features from the base model
                outputs = self.base_model(x)
                features = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
                return features

        # Initialize the modified model
        self.model = PhikonEmbedder(self.base_model, self.num_features)

        self.model.to('cuda')
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'phikon',
            'kwargs': {}
        }
@register_torch
class gigapath(TorchFeatureExtractor):
    """
    Prov-GigaPath feature extractor, with Vision Transformer backbone

    Parameters    
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'gigapath'

    def __init__(self, tile_px=256):
        super().__init__()

        local_dir = WEIGHTS_DIR
        model_name = "gigapath.bin"
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
        
        model_path = os.path.join(local_dir, model_name)

        if not os.path.exists(model_path):
            self.model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            torch.save(self.model.state_dict(), model_path)
        else:
            self.model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)

        self.model.to('cuda')
        self.num_features = 1024
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'gigapath',
            'kwargs': {}
        }

@register_torch
class kaiko_s8(TorchFeatureExtractor):
    """
    Kaiko S8 feature extractor, with small Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'kaiko_s8'

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vits8', trust_repo=True)

        self.model.to('cuda')
        self.num_features = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'kaiko_s8',
            'kwargs': {}
        }

@register_torch
class kaiko_s16(TorchFeatureExtractor):
    """
    Kaiko S16 feature extractor, with small Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'kaiko_s16'

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vits16', trust_repo=True)

        self.model.to('cuda')
        self.num_features = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'kaiko_s16',
            'kwargs': {}
        }

@register_torch
class kaiko_b8(TorchFeatureExtractor):
    """
    Kaiko B8 feature extractor, with base Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'kaiko_b8'

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vitb8', trust_repo=True)
        self.model.to('cuda')
        self.num_features = 768
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'kaiko_b8',
            'kwargs': {}
        }

@register_torch
class kaiko_b16(TorchFeatureExtractor):
    """
    Kaiko B16 feature extractor, with base Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'kaiko_b16'

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vitb16', trust_repo=True)
        self.model.to('cuda')
        self.num_features = 768
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'kaiko_b16',
            'kwargs': {}
        }


@register_torch
class kaiko_l14(TorchFeatureExtractor):
    """
    Kaiko L14 feature extractor, with large Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'kaiko_l14'

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vitl14', trust_repo=True)

        self.model.to('cuda')
        self.num_features = 1024
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'kaiko_l14',
            'kwargs': {}
        }

#TODO: Implement CONCH
"""
@register_torch
class conch(TorchFeatureExtractor):
    def __init__(self, tile_px=256):
        super().__init__()
        self.model = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=keys['conch'])
        self.model.to('cuda')
        self.num_features = 1024
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # Transform to float tensor
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}
"""

class VisionTransformerMoCoWithoutHead(VisionTransformer):
    """
    Vision Transformer model with MoCo pretraining, without the final fully connected layer

    Parameters
    ----------
    *args : list
        Variable length argument list
    **kwargs : dict
        Keyword arguments
    pretext_token : bool
        Whether to add a pretext token
    stop_grad_conv1 : bool
        Whether to stop gradient for the first convolutional layer
    
    Attributes
    ----------
    num_prefix_tokens : int
        The number of prefix tokens
    pretext_token : nn.Parameter
        The pretext token
    embed_len : int
        The length of the embedding
    pos_embed : nn.Parameter
        The positional embedding
    embed_dim : int
        The embedding dimension
    patch_embed : PatchEmbed
        The patch embedding layer
    pos_drop : nn.Dropout
        The positional dropout layer
    blocks : nn.Sequential
        The transformer blocks
    norm : nn.LayerNorm
        The layer normalization layer
    cls_token : nn.Parameter
        The class token
    dist_token : nn.Parameter
        The distance token
    norm_pre : nn.LayerNorm
        The layer normalization layer for the prefix tokens
    
    Methods
    -------
    _pos_embed(x)
        Add positional embedding to the input tensor
    _ref_embed(ref)
        Add positional embedding to the reference tensor
    _pos_embed_with_ref(x, ref)
        Add positional embedding to the input tensor with reference
    forward_features(x, ref)
        Forward pass of the model for features
    forward(x, ref)
        Forward pass of the model
    build_2d_sincos_position_embedding()
        Build 2D sin-cos position embedding
    """
    def __init__(self, pretext_token=True, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # inserting a new token
        self.num_prefix_tokens += (1 if pretext_token else 0)
        self.pretext_token = nn.Parameter(torch.ones(1, 1, self.embed_dim)) if pretext_token else None
        embed_len = self.patch_embed.num_patches if self.no_embed_class else self.patch_embed.num_patches + 1
        embed_len += 1 if pretext_token else 0
        self.embed_len = embed_len

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pretext_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token', 'pretext_token'}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)
    
    def _ref_embed(self, ref):
        B, C, H, W = ref.shape
        ref = self.patch_embed.proj(ref)
        if self.patch_embed.flatten:
            ref = ref.flatten(2).transpose(1, 2)  # BCHW -> BNC
        ref = self.patch_embed.norm(ref)
        return ref

    def _pos_embed_with_ref(self, x, ref):
        pretext_tokens = self.pretext_token.expand(x.shape[0], -1, -1) * 0 + ref
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, ref=None):
        x = self.patch_embed(x)
        if ref is None:
            x = self._pos_embed(x)
        else:
            ref = self._ref_embed(ref).mean(dim=1, keepdim=True)
            x = self._pos_embed_with_ref(x, ref)
        # x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, ref=None):
        x_out = self.forward_features(x, ref)
        #Average the features to reduce the dimensionality
        x_out = x_out.mean(dim=1)
        return x_out

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_prefix_tokens == 2, 'Assuming two and only two tokens, [pretext][cls]'
        pe_token = torch.zeros([1, 2, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
        
@register_torch
class pathoduet_he(TorchFeatureExtractor):
    """
    PathoDuet HE-trained feature extractor, with Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformerMoCoWithoutHead   
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict   
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'pathoduet_he'

    def __init__(self, tile_px=256):
        super().__init__()
        self.model = VisionTransformerMoCoWithoutHead(pretext_token=True, global_pool='avg')
        checkpoint = torch.load(f'{WEIGHTS_DIR}/checkpoint_HE.pth')
        # Remove the 'head.weight' and 'head.bias' keys from the state dictionary
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('head.')}

        # Load the modified state dictionary into your model
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to('cuda')
        self.num_features = 768
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'pathoduet_he',
            'kwargs': {}
        }

@register_torch
class pathoduet_ihc(TorchFeatureExtractor):
    """
    PathoDuet IHC-trained feature extractor, with Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformerMoCoWithoutHead   
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict   
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'pathoduet_ihc'

    def __init__(self, tile_px=256):
        super().__init__()
        self.model = VisionTransformerMoCoWithoutHead(pretext_token=True, global_pool='avg')
        checkpoint = torch.load(f'{WEIGHTS_DIR}/checkpoint_IHC.pth')
        # Remove the 'head.weight' and 'head.bias' keys from the state dictionary
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('head.')}
        # Load the modified state dictionary into your model
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to('cuda')
        self.num_features = 768
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}


    def dump_config(self):
        return {
            'class': 'pathoduet_ihc',
            'kwargs': {}
        }

@register_torch
class hibou_b(TorchFeatureExtractor):
    """
    Hibou B feature extractor, with base Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : HibouEmbedder
        The Hibou model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'hibou_b'

    def __init__(self, tile_px=256):
        super().__init__()
        
        model_name = "hibou_b"
        self.num_features = 768
        self.processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
        base_model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        
        # Add a new head to include adaptive average pooling
        class HibouEmbedder(nn.Module):
            """
            Hibou Embedder model, which adds mean pooling to the base model.

            Parameters
            ----------
            base_model : nn.Module
                The base Hibou-B model
            
            Methods
            -------
            forward(x)
                Forward pass of the model
            """
            def __init__(self, base_model):
                super(HibouEmbedder, self).__init__()
                self.base_model = base_model
                

            def forward(self, x):
                # Apply adaptive average pooling
                x = self.base_model(x)
                x = x[0]
                x = torch.mean(x, dim=1)
                return x

        self.model = HibouEmbedder(base_model)
        self.model.to('cuda')
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'hibou_b',
            'kwargs': {}
        }

@register_torch
class virchow(TorchFeatureExtractor):
    """
    Virchow feature extractor, with Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile

    Attributes
    ---------- 
    model : VirchowEmbedder
        The Virchow model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict   
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'virchow'

    def __init__(self, tile_px=256):
        super().__init__()
        local_dir = WEIGHTS_DIR
        base_model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)
        base_model.to('cuda')

        # Modify the classifier to output the concatenated embeddings
        class VirchowEmbedder(nn.Module):
            """
            Virchow Embedder model, which concatenates the class token and average pool of patch tokens.

            Methods
            -------
            forward(x)
                Forward pass of the model
            """
            def __init__(self):
                super(VirchowEmbedder, self).__init__()

            def forward(self, x):
                x = base_model(x)
                cls_token = x[:, 0]
                patch_tokens = x[:, 1:]
                avg_pool = patch_tokens.mean(dim=1)
                # Concatenate class token and average pool of patch tokens
                embedding = torch.cat((cls_token, avg_pool), dim=-1)
                return embedding

        self.model = VirchowEmbedder()

        self.model.to('cuda')
        self.num_features = 2560  # Update the number of features to reflect the new embedding size
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'virchow',
            'kwargs': {}
        }


@register_torch
class h_optimus_0(TorchFeatureExtractor):
    """
    H-Optimus feature extractor, with large Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'h_optimus_0'

    def __init__(self, tile_px=256):
        super().__init__()

        params = {
            'patch_size': 14, 
            'embed_dim': 1536, 
            'depth': 40, 
            'num_heads': 24, 
            'init_values': 1e-05, 
            'mlp_ratio': 5.33334, 
            'mlp_layer': functools.partial(
                timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
            ), 
            'act_layer': torch.nn.modules.activation.SiLU, 
            'reg_tokens': 4, 
            'no_embed_class': True, 
            'img_size': 224, 
            'num_classes': 0, 
            'in_chans': 3
        }

        self.model = timm.models.VisionTransformer(**params)
        self.model.load_state_dict(torch.load(f"{WEIGHTS_DIR}/h_optimus_0.pth"))
        self.model.to('cuda')
        self.num_features = 768
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'h_optimus_0',
            'kwargs': {}
        }


@register_torch
class transpath_mocov3(TorchFeatureExtractor):
    """
    TransPath MoCoV3 feature extractor, with small Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'transpath_mocov3'

    def __init__(self, tile_px=256):
        super().__init__()
        self.model = timm.create_model(
            model_name="hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3",
            pretrained=True,
            num_heads=12,
            ).eval()
        self.model.to('cuda')
        self.num_features = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'transpath_mocov3',
            'kwargs': {}
        }


#TODO: Correct implementation of the model, does not work right now.
@register_torch
class beph(TorchFeatureExtractor):
    """
    BEPH feature extractor, with base Vision Transformer backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    
    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = 'beph'

    def __init__(self, tile_px=256):
        super().__init__()
        state_dict = torch.load(f"{WEIGHTS_DIR}/BEPH_backbone.pth")['state_dict']
        self.model = timm.create_model('beitv2_base_patch16_224.in1k_ft_in22k_in1k', pretrained=True, num_classes=0)
        self.model.to('cuda')
        #Load the state dictionary
        self.model.load_state_dict(state_dict)

        self.num_features = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'beph',
            'kwargs': {}
        }

