<div style="text-align: center;">
 <img src="thumbnail_PathBench-logo-horizontaal.png" alt="PathBench" width="800" height="200">
</div>

# PathBench
PathBench is a Python package designed to facilitate benchmarking, experimentation, and optimization of Multiple Instance Learning (MIL) for computational histopathology. It provides tools for conducting experiments, evaluating performance, visualizing results, and optimizing hyperparameters. PathBench is built on top of SlideFlow for handling Whole Slide images and integrates Optuna for hyperparameter optimization. PathBench is useful for researchers aiming to benchmarking different pipeline parameters / models for use in their subdomain (in which case the benchmark mode is more suitable) and users starting a Computational Pathology project and wanting to find a suitable pipeline architecture (in which case the optimization mode is more suitable).

PathBench operates in two modes: Benchmark-mode and Optimization-mode. Benchmark mode takes in different options for the computational pipeline (e.g. normalization methods, feature extractors) and benchmarks all possible combinations, outputting a performance table sorted on mode performance. Optimization mode simply aims to find the optimal set of computational pipeline hyperparameters to optimize a set performance objective, it will not test all possible combinations.

One can use PathBench for binary classification, multiclass classification, regression and survival predicition problems. Multiple datasets can be integrated into a single experiment to allowing for training or testing on different data sources. All user parameters are captured in a single .yaml file.

The output of benchmarking experiments can be loaded into a plotly-based visualization app for further investigation. We also provide a pipeline which calculates the semantic feature similarity between feature bags which can then be used to build model ensembles.

PathBench is being developed at the Leiden University Medical Center: Department of Pathology.
- Lead Developer: Siemen Brussee
- Developers: 

## Features

- Benchmarking w.r.t.
    - Tile sizes, magnifications (e.g. 256px, 20x)
    - Normalization methods (e.g. Macenko, Reinhard)
    - Feature extractors (e.g. UNI, GigaPath)
    - MIL aggregators (e.g. CLAM, DSMIL)
- Interpretable visualizations of benchmark output
- Plotly-based benchmark visualization tool
- Efficient Tile processing and QC pipeline inherited by Slideflow
- Optuna-based optimization w.r.t. the benchmark parameters, to quickly find good candidate solutions.

## Package Structure

- pathbench/
  - pathbench/
    - benchmarking/
      - benchmark.py
    - experiment/
      - experiment.py
    - utils
      - utils.py
  - requirements.txt
  - README.md
  - LICENSE
  - setup.py

## Installation

You can install PathBench and its dependencies using pip:

```bash
pip install pathbench
```
## Normalization
PathBench currently supports:
- Macenko
- Reinhard
- CycleGan
- Ruifrok

normalization.
## Feature Extractors
PathBench supports a wide range of different feature extractors, including SOTA foundation models for pathology. Most of these models are automatically downloaded by PathBench, however, some models require a huggingface account key to access the model (labeled 'Gated' in the feature extraction table) or require manually downloading the model weights (labeled 'Manual' in the extraction table). For each of the models, a link to the publication is given, and for the manual/gated models, the link for downloading the models or gaining model access are also provided.

| Feature Extractor | Acquisition | Link |
|----------|----------|----------|
| ImageNet-ResNet50 | Automatic | NA | 
| CTransPath | Automatic | [Link](https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file) |
| MoCoV3-TransPath | Automatic | [Link](https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file) |
| HistoSSL | Automatic | [Link](https://github.com/owkin/HistoSSLscaling) |
| RetCCL | Automatic | [Link](https://github.com/Xiyue-Wang/RetCCL) |
| PLIP | Automatic | [Link](https://github.com/PathologyFoundation/plip?tab=readme-ov-file) |
| Lunit DINO | Automatic | [Link](https://github.com/lunit-io/benchmark-ssl-pathology) |
| Lunit SwAV | Automatic | [Link](https://github.com/lunit-io/benchmark-ssl-pathology)  |
| Lunit Barlow Twins | Automatic | [Link](https://github.com/lunit-io/benchmark-ssl-pathology) |
| Lunit MocoV2 | Automatic | [Link](https://github.com/lunit-io/benchmark-ssl-pathology)  |
| Phikon | Automatic | [Link](https://huggingface.co/owkin/phikon) |
| PathoDuet-HE | Manual | [Link](https://github.com/openmedlab/PathoDuet) [Weights](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM)|
| PathoDuet-IHC | Manual | [Link](https://github.com/openmedlab/PathoDuet) [Weights](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM)|
| Virchow | Gated | [Link](https://huggingface.co/paige-ai/Virchow)|
| Hibou-B | Automatic | [Link](https://huggingface.co/histai/hibou-b) |
| UNI | Gated | [Link](https://huggingface.co/MahmoodLab/UNI) |
| Prov-GigaPath | Gated | [Link](https://huggingface.co/prov-gigapath/prov-gigapath) |
| Kaiko-S8 | Automatic | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| Kaiko-S16 | Automatic | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| Kaiko-B8 | Automatic | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| Kaiko-B16 | Automatic | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| Kaiko-L14 | Automatic | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| H-Optimus-0 | Automatic | [Link](https://huggingface.co/bioptimus/H-optimus-0) |

## MIL aggregators
In addition to a wide range of feature extractors, PathBench also includes a wide variety of MIL aggregation methods. Most of these support all tasks (Binary classification, Muliclass classifcation, regression and survival prediction), but some like the CLAM-models only support binary classification. We are actively working on extending support for these models.

| MIL aggregator | Bin. class. | Multi-class. | Regression | Survival | Link |
|----------|----------|----------|----------|----------|----------|
| CLAM-SB | Supported | Not Supported | Not Supported | Not Supported | [Link](https://github.com/mahmoodlab/CLAM) |
| CLAM-MB | Supported | Not Supported | Not Supported | Not Supported | [Link](https://github.com/mahmoodlab/CLAM)  |
| Attention MIL | Supported | Supported | Supported | Supported | [Link](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)|
| TransMIL | Supported | Supported | Supported | Supported | [Link](https://github.com/szc19990412/TransMIL) |
| HistoBistro Transformer | Supported | Supported | Supported | Supported | [Link](https://github.com/peng-lab/HistoBistro) |
| Linear MIL | Supported | Supported | Supported | Supported | NA |
| Mean MIL | Supported | Supported | Supported | Supported | NA |
| Max MIL | Supported | Supported | Supported | Supported | NA |
| Log-Sum-Exp MIL | Supported | Supported | Supported | Supported | NA |
| LSTM-MIL | Supported | Supported | Supported | Supported | NA |
| DeepSet-MIL | Supported | Supported | Supported | Supported |[Link](https://github.com/manzilzaheer/DeepSets)|
| Distribution-pool MIL | Supported | Supported | Supported | Supported | NA |
| VarMIL | Supported | Supported | Supported | Supported | [Link](https://github.com/NKI-AI/dlup-lightning-mil)|
| DSMIL | Supported | Supported | Supported | Supported | [Link](https://github.com/binli123/dsmil-wsi)  |

# PathBench Configuration Example

To use PathBench, you need to provide a configuration file in YAML format. Below is an example configuration file:

```yaml
experiment:
  project_name: Example_Project # Name of the project, where the results will be saved
  annotation_file: /path/to/your/annotation_file.csv # Path to the annotation file
  #splits: /path/to/your/split_file.json # Path to the split file, not required.
  balancing: category # Training set balancing strategy, can be None, category, slide, patient or tile. 
  split_technique: k-fold # Splitting technique, can be k-fold or fixed
  epochs: 5 # Number of training epochs
  batch_size: 32 # Batch size
  k: 2 # Number of folds, if split-technique is k-fold
  val_fraction: 0.1 # Fraction of the training set used for validation
  aggregation_level: slide # Aggregation level, can be slide or patient
  with_continue: True # Continue training from a previous checkpoint, if available
  task: classification # Task, can be classification, regression or survival
  weights_dir: /path/to/your/pretrained_weights # Path to the model weights
  visualization: # Visualization options, options: learning_curve, confusion_matrix, roc_curve, umap, mosaic
    - learning_curve
    - confusion_matrix
    - roc_curve
    - umap
    - mosaic
  mode: optimization # Mode to use, either benchmark or optimization

optimization:
  objective_metric: balanced_accuracy # Objective metric to optimize
  sampler: TPESampler # Algorithm to use for optimization: grid_search, TPE, Bayesian
  trials: 100 # Number of optimization trials
  pruner: HyperbandPruner

datasets: # List of datasets to use, each dataset should have a name, slide_path, tfrecord_path, tile_path and used_for.
  - name: dataset_1
    slide_path: /path/to/your/dataset_1/slides
    tfrecord_path: /path/to/your/dataset_1/tfrecords
    tile_path: /path/to/your/dataset_1/tiles
    used_for: training

  - name: dataset_2
    slide_path: /path/to/your/dataset_2/slides
    tfrecord_path: /path/to/your/dataset_2/tfrecords
    tile_path: /path/to/your/dataset_2/tiles
    used_for: testing

benchmark_parameters: # Parameters for the benchmarking, can be used to compare different methods
  tile_px: # Tile size in pixels
    - 256
  tile_um: # Tile size in micrometers
    - 20x
  normalization: # Normalization method, can be macenko, reinhard, ruifrok or cyclegan
    - macenko
    - reinhard
  feature_extraction: # Feature extraction methods
    - resnet50_imagenet
    - hibou_b
  mil: # Multiple instance learning aggregation methods
    - Attention_MIL
    - dsmil

# Available normalization methods:
# - macenko
# - reinhard
# - ruifrok
# - cyclegan

# Available feature extraction methods:
# - resnet50_imagenet
# - CTransPath
# - transpath_mocov3
# - RetCCL
# - PLIP
# - HistoSSL
# - uni
# - dino
# - mocov2
# - swav
# - phikon
# - gigapath
# - barlow_twins
# - hibou_b
# - pathoduet_ihc
# - pathoduet_he
# - kaiko_s8
# - kaiko_s16
# - kaiko_b8
# - kaiko_b16
# - kaiko_l14
# - h_optimus_0
# - virchow

# Available MIL aggregation methods:
# - CLAM_SB
# - CLAM_MB
# - Attention_MIL
# - transmil
# - bistro.transformer
# - linear_mil
# - mean_mil
# - max_mil
# - lse_mil
# - lstm_mil
# - deepset_mil
# - distributionpooling_mil
# - dsmil
# - varmil
```

## Extending PathBench
PathBench is designed such that it easy to add new feature extractors and MIL aggregation models. New feature extractors are added to pathbench/models/feature_extractors.py and follow this format:
```python
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
```
Feature extractor models require a @register_torch descriptor to be recognizable by PathBench as a specifiable feature extractor. Furthermore, the class requires a model to be specified, the embedding size to be specified and a transformation pipeline. For more information, please see the [slideflow documentation](https://slideflow.dev/).

Adding MIL aggregation methods is done similarly, but in the pathbench/models/aggregators.py script:
```python
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
```
The MIL aggregation function should take the bags in its forward function and output the prediction scores. Typically, this involves an encoder network and a prediction head. If the aggregation method supports attention values, one can add an additional method:
```python
    def calculate_attention(self, bags, lens, apply_softmax=None):
        embeddings = self.encoder(bags)
        attention_scores = self.attention(embeddings)
        if apply_softmax:
            attention_scores = F.softmax(attention_scores, dim=1)
        return attention_scores
```
Which calculates attention for the input bags. This can then be used to generate attention heatmaps.

## Futher extension
For more fundamental changes, and adding new normalization methods, one needs to change the underlying slideflow code. PathBench uses a [forked version](https://github.com/Sbrussee/slideflow_fork) of the slideflow code, which needs to be changed in order to implement these major changes.
<div style="text-align: center;">
 <img src="PathBench-logo-gecentreerd.png" alt="PathBench Logo" width="550" height="400">
</div>
