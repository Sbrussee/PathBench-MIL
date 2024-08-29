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
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
import os
import yaml
import math
import functools
from functools import reduce
from operator import mul


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
    https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html
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
    

#NEEDS KEY
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
        local_dir = WEIGHTS_DIR
        model_name = "uni.bin"
        model_temp_name = "pytorch_model.bin"
        model_path = os.path.join(local_dir, model_name)

        if not os.path.exists(model_path):
            temp_model_path = hf_hub_download(repo_id="MahmoodLab/UNI", filename=model_temp_name, local_dir=local_dir, force_download=True)
            os.rename(temp_model_path, model_path)
        
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
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

#NEEDS KEY
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
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
        
#NEEDS LOCAL WEIGHTS
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


#NEEDS LOCAL WEIGHTS
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
                transforms.Normalize(mean=(0.7068, 0.5655, 0.722), std=(0.195, 0.2316, 0.1816)),
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


#GATED
@register_torch
class hibou_l(TorchFeatureExtractor):
    """
    Hibou L feature extractor, with large Vision Transformer backbone

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
    tag = 'hibou_l'

    def __init__(self, tile_px=256):
        super().__init__()
        
        model_name = "hibou_l"
        self.num_features = 768
        self.processor = AutoImageProcessor.from_pretrained("histai/hibou-l", trust_remote_code=True)
        base_model = AutoModel.from_pretrained("histai/hibou-l", trust_remote_code=True)
        
        # Add a new head to include adaptive average pooling
        class HibouEmbedder(nn.Module):
            """
            Hibou Embedder model, which adds mean pooling to the base model.

            Parameters
            ----------
            base_model : nn.Module
                The base Hibou-L model
            
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
                transforms.Normalize(mean=(0.7068, 0.5655, 0.722), std=(0.195, 0.2316, 0.1816)),
            ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'hibou_l',
            'kwargs': {}
        }
#GATED
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
                self.base_model = base_model.cuda()

            def forward(self, x):
                x = self.base_model(x)
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
class virchow2(TorchFeatureExtractor):
    """
    Virchow2 feature extractor, with Huge Vision Transformer backbone.
    """
    tag = 'virchow2'

    def __init__(self, tile_px=256):
        super().__init__()
        local_dir = WEIGHTS_DIR
        base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)

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
                self.base_model = base_model.cuda()

            def forward(self, x):
                x = self.base_model(x)
                cls_token = x[:, 0]
                patch_tokens = x[:, 5:]
                avg_pool = patch_tokens.mean(dim=1)
                # Concatenate class token and average pool of patch tokens
                embedding = torch.cat((cls_token, avg_pool), dim=-1)
                return embedding

        self.model = VirchowEmbedder()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]  
        )
        self.model.to('cuda')
        self.num_features = 1280
        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'virchow2',
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

        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        self.model.to("cuda")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(224),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
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

@register_torch
class exaone_path(TorchFeatureExtractor):
    """
    EXAONE Path feature extractor, with Vision Transformer backbone

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
    tag = 'exaone_path'

    def __init__(self, tile_px=256):
        super().__init__()

        def _no_grad_trunc_normal_(tensor, mean, std, a, b):
            # Cut & paste from PyTorch official master until it's in a few official releases - RW
            # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
            def norm_cdf(x):
                # Computes standard normal cumulative distribution function
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                            "The distribution of values may be incorrect.",
                            stacklevel=2)

            with torch.no_grad():
                # Values are generated by using a truncated uniform distribution and
                # then using the inverse CDF for the normal distribution.
                # Get upper and lower cdf values
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)

                # Uniformly fill tensor with values from [l, u], then translate to
                # [2l-1, 2u-1].
                tensor.uniform_(2 * l - 1, 2 * u - 1)

                # Use inverse cdf transform for normal distribution to get truncated
                # standard normal
                tensor.erfinv_()

                # Transform to proper mean, std
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)

                # Clamp to ensure it's in the proper range
                tensor.clamp_(min=a, max=b)
                return tensor


        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
            return _no_grad_trunc_normal_(tensor, mean, std, a, b)


        def drop_path(x, drop_prob: float = 0., training: bool = False):
            if drop_prob == 0. or not training:
                return x
            keep_prob = 1 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            output = x.div(keep_prob) * random_tensor
            return output


        class DropPath(nn.Module):
            """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
            """
            def __init__(self, drop_prob=None):
                super(DropPath, self).__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return drop_path(x, self.drop_prob, self.training)


        class Mlp(nn.Module):
            def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.act = act_layer()
                self.fc2 = nn.Linear(hidden_features, out_features)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.drop(x)
                return x


        class Attention(nn.Module):
            def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
                super().__init__()
                self.num_heads = num_heads
                head_dim = dim // num_heads
                self.scale = qk_scale or head_dim ** -0.5

                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
                self.attn_drop = nn.Dropout(attn_drop)
                self.proj = nn.Linear(dim, dim)
                self.proj_drop = nn.Dropout(proj_drop)

            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x, attn


        class Block(nn.Module):
            def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
                super().__init__()
                self.norm1 = norm_layer(dim)
                self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.norm2 = norm_layer(dim)
                mlp_hidden_dim = int(dim * mlp_ratio)
                self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

            def forward(self, x, return_attention=False):
                y, attn = self.attn(self.norm1(x))
                if return_attention:
                    return attn
                x = x + self.drop_path(y)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x


        class PatchEmbed(nn.Module):
            """ Image to Patch Embedding
            """
            def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
                super().__init__()
                num_patches = (img_size // patch_size) * (img_size // patch_size)
                self.img_size = img_size
                self.patch_size = patch_size
                self.num_patches = num_patches

                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

            def forward(self, x):
                B, C, H, W = x.shape
                x = self.proj(x).flatten(2).transpose(1, 2)
                return x
            
        class VisionTransformer(nn.Module, PyTorchModelHubMixin):
            def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
                super().__init__()
                self.num_features = self.embed_dim = embed_dim

                self.patch_embed = PatchEmbed(
                    img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
                num_patches = self.patch_embed.num_patches

                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.pos_drop = nn.Dropout(p=drop_rate)

                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
                self.blocks = nn.ModuleList([
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depth)])
                self.norm = norm_layer(embed_dim)

                # Classifier head
                self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

                trunc_normal_(self.pos_embed, std=.02)
                trunc_normal_(self.cls_token, std=.02)
                self.apply(self._init_weights)

                # config for PyTorchModelHubMixin
                self.config = {
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "in_chans": in_chans,
                    "num_classes": num_classes,
                    "embed_dim": embed_dim,
                    "depth": depth,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                    "qkv_bias": qkv_bias,
                    "qk_scale": qk_scale,
                    "drop_rate": drop_rate,
                    "attn_drop_rate": attn_drop_rate,
                    "drop_path_rate": drop_path_rate,
                }

            def _init_weights(self, m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            def interpolate_pos_encoding(self, x, w, h):
                npatch = x.shape[1] - 1
                N = self.pos_embed.shape[1] - 1
                if npatch == N and w == h:
                    return self.pos_embed
                class_pos_embed = self.pos_embed[:, 0]
                patch_pos_embed = self.pos_embed[:, 1:]
                dim = x.shape[-1]
                w0 = w // self.patch_embed.patch_size
                h0 = h // self.patch_embed.patch_size
                # we add a small number to avoid floating point error in the interpolation
                # see discussion at https://github.com/facebookresearch/dino/issues/8
                w0, h0 = w0 + 0.1, h0 + 0.1
                patch_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                    mode='bicubic',
                )
                assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

            def prepare_tokens(self, x):
                B, nc, w, h = x.shape
                x = self.patch_embed(x)  # patch linear embedding

                # add the [CLS] token to the embed patch tokens
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                # add positional encoding to each token
                x = x + self.interpolate_pos_encoding(x, w, h)

                return self.pos_drop(x)

            def forward(self, x):
                x = self.prepare_tokens(x)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                # print(x.type())
                return x[:, 0]

            def get_last_selfattention(self, x):
                x = self.prepare_tokens(x)
                for i, blk in enumerate(self.blocks):
                    if i < len(self.blocks) - 1:
                        x = blk(x)
                    else:
                        # return attention of the last block
                        return blk(x, return_attention=True)

            def get_intermediate_layers(self, x, n=1):
                x = self.prepare_tokens(x)
                # we return the output tokens from the `n` last blocks
                output = []
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    if len(self.blocks) - i <= n:
                        output.append(self.norm(x))
                return output

        self.model = VisionTransformer().from_pretrained("LGAI-EXAONE/EXAONEPath").eval()
        self.num_features = 768
        self.transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        )
        self.model.to('cuda')
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'exaone_path',
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

