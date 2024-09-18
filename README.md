<p align="center">
 <img src="thumbnail_PathBench-logo-horizontaal.png" alt="PathBench" width="800" height="200">
</p>

# PathBench-MIL: A comprehensive, flexible Benchmarking/AutoML framework for multiple instance learning in pathology.

PATHBENCH IS CURRENTLY UNDER DEVELOPMENT, SOME FEATURES MAY NOT WORK AS INTENDED

PathBench is a Python package designed to facilitate benchmarking, experimentation, and optimization of Multiple Instance Learning (MIL) for computational histopathology. It provides tools for conducting experiments, evaluating performance, visualizing results, and optimizing hyperparameters. PathBench is built on top of the excellent [SlideFlow](https://github.com/jamesdolezal/slideflow) package for handling Whole Slide images and integrates Optuna for hyperparameter optimization. PathBench is useful for researchers aiming to benchmarking different pipeline parameters / models for use in their subdomain (in which case the benchmark mode is more suitable) and users starting a Computational Pathology project and wanting to find a suitable pipeline architecture (in which case the optimization mode is more suitable).

PathBench operates in two modes: Benchmark-mode and Optimization-mode. Benchmark mode takes in different options for the computational pipeline (e.g. normalization methods, feature extractors) and benchmarks all possible combinations, outputting a performance table sorted on mode performance. Optimization mode simply aims to find the optimal set of computational pipeline hyperparameters to optimize a set performance objective, it will not test all possible combinations.

One can use PathBench for binary classification, multiclass classification, regression and survival predicition problems. Multiple datasets can be integrated into a single experiment to allowing for training or testing on different data sources. All user parameters are captured in a single .yaml file.

The output of benchmarking experiments can be loaded into a plotly-based visualization app for further investigation. We also provide a pipeline which calculates the semantic feature similarity between feature bags which can then be used to build model ensembles.

PathBench is being developed at the Leiden University Medical Center: Department of Pathology.
- Lead Developer: Siemen Brussee
- Developers:

PathBench's documentation is available [here](https://pathbench.readthedocs.io/en/latest/index.html)

# Installation Guide for PathBench

## Prerequisites

- Python 3.8
- Git

## Steps to Install PathBench and SlideFlow Fork

1. **Clone the Repository:**

    ```bash
    git clone --recurse_submodules https://github.com/sbrussee/PathBench.git
    cd PathBench
    ```

2. **Run `setup_pathbench.py`:**

    Run the existing script to set up the virtual environment and install necessary tools.

    ```bash
    python setup_pathbench.py
    ```

    This script will:
    - Create a virtual environment named `pathbench_env`.
    - Upgrade `pip` and install `setuptools`, `wheel`, and `versioneer`.

3. **Activate the Virtual Environment:**

    - macOS/Linux:
        ```bash
        source pathbench_env/bin/activate
        ```
    - Windows:
        ```bash
        pathbench_env\Scripts\activate
        ```
4. **Install `slideflow` Package:**

    Navigate to the `slideflow_fork` directory and install it:

    ```bash
    cd ../slideflow_fork
    pip install -e .
    ```

    Or, if you do not need to modify the code:

    ```bash
    pip install .
    ```
    
5. **Install `pathbench-mil` Package:**

    After activating the virtual environment, install the `pathbench-mil` package.

    ```bash
    pip install -e .
    ```

    Or, if you do not need to modify the code:

    ```bash
    pip install .
    ```


# PathBench Configuration Example
To use PathBench, you need to provide a configuration file in YAML format. Below is an example configuration file:

```yaml
experiment:
  project_name: Example_Project # Name of the project, where the results will be saved
  annotation_file: /path/to/your/annotation_file.csv # Path to the annotation file
  balancing: category # Training set balancing strategy, can be None, category, slide, patient or tile.
  num_workers: 0 # Number of workers for data loading, 0 for no parallelization.
  split_technique: k-fold # Splitting technique, can be k-fold or fixed
  epochs: 5 # Number of training epochs
  batch_size: 32 # Batch size
  bag_size : 512 # Bag size for MIL
  k: 2 # Number of folds, if split-technique is k-fold
  val_fraction: 0.1 # Fraction of the training set used for validation
  best_epoch_based_on: val_loss # Metric to be used for selecting the best epoch, can be val_loss or any metric in evaluation
  aggregation_level: slide # Aggregation level, can be slide or patient
  with_continue: True # Continue training from a previous checkpoint, if available
  task: classification # Task, can be classification, regression or survival
  visualization: # Visualization options, options: CLASSIFICATION: confusion_matrix, precision_recall_curve, roc_curve, top_tiles URVIVAL: survival_roc, concordance_index, calibration REGRESSION: predicted_vs_actual, residuals, qq
    - learning_curve
    - confusion_matrix
    - roc_curve
  mode: optimization # Mode to use, either benchmark or optimization

  custom_metrics: [RocAuc]: List of evaluation metrics to measure in the validation set during model training. Needs to be either specified in metrics.py or a fastai metric: https://docs.fast.ai/metrics.html

  qc: # List of quality control methods to be used, supports Otsu, Gaussian, GaussianV2 and Otsu-CLAHE
    - GaussianV2 # Faster version of Gaussian blur tissue detection
    - Otsu-CLAHE # Otsu thresholding tissue detection with CLAHE-enhanced V-channel in HSV space for increased contrast.

  qc_filters: #Tile-level filters for discarding tiles
    - grayspace_threshold : 0.05 #Pixels below this value (ranged 0-1) are considered gray.
    - grayspace_fraction: 0.6 # Image tiles with grayspace above this fraction are discarded.
    - whitespace_threshold: 230 #Pixel intensities (0-255) above this value are considered white.
    - whitespace_fraction: 1.0 # Image tiles with whitespace above this fraction are discarded.

optimization:
  objective_metric: balanced_accuracy # Objective metric to optimize
  objective_mode: max # Optimization mode, can be 'max' or 'min'
  objective_dataset: test # Dataset to be used for the objective metric, can be 'val' or 'test'
  sampler: TPESampler # Algorithm to use for optimization: grid_search, TPE, Bayesian
  trials: 100 # Number of optimization trials
  pruner: HyperbandPruner # Pruner for optimization, can be Hyperband, Median etc.

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

  loss: # Loss functions, as specified in losses.py
    - CrossEntropyLoss

  augmentation: # MIL-friendly augmentations, as specified in augmentations.py
    - patch_mixing

  activation_function: # activation function for the MIL encoder, supports any pytorch.nn activation function.
    - ReLU

# Available normalization methods:
# - macenko
# - reinhard
# - cyclegan

# Available feature extraction methods:
# - resnet50_imagenet
# - CTransPath
# - transpath_mocov3
# - RetCCL
# - PLIP
# - HistoSSL
# - uni
# - conch
# - dino
# - mocov2
# - swav
# - phikon
# - phikon_v2
# - gigapath
# - barlow_twins
# - hibou_b
# - hibou_l
# - pathoduet_ihc
# - pathoduet_he
# - kaiko_s8
# - kaiko_s16
# - kaiko_b8
# - kaiko_b16
# - kaiko_l14
# - h_optimus_0
# - virchow
# - virchow2

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
# - perceiver_mil
# - air_mil

#Available Loss functions
  # - Classification:
     # - CrossEntropyLoss
     # - FocalLoss
     # - LabelSmoothingCrossEntropyLoss
     # - CrossEntropyWithEntropyMinimizationLoss
     # - AttentionEntropyMinimizedCrossEntropyLoss
     # - DiversityRegularizedCrossEntropyLoss
     # - SparseAttentionCrossEntropyLoss

#Available MIL-friendly augmentations:
  # - patch_dropout
  # - add_gaussian_noise
  # - random_scaling
  # - feature_masking
  # - feature_dropout
  # - patchwise_scaling
  # - feature_permutation
  # - patch_mixing
  # - cutmix

weights_dir : ./pretrained_weights # Path to the model weights, and where newly retrieved model weights will be saved
hf_key: YOUR_HUGGINGFACE_TOKEN # Token for Hugging Face model hub to access gated models, if you do not have one, just set to None
```

# Setting up a Project
PathBench inherits the project functionality from SlideFlow. PathBench allows creating projects through the configuration file. In the configuration, project-specific settings can be specified in the `experiment` section. The `experiment` section also saves several other important settings:

### Project-related settings:

- `project_name`: The name of the project.
- `annotation_file`: The path to the annotation file.

### Training-related settings:

- `balancing`: The balancing technique to be used. This can be `tile`, `slide`, `patient`, or `category`. Balancing is used to construct training batches with a balanced distribution of classes/patients/slides/tiles.
- `split_technique`: The split technique to be used. This can be `k-fold` or `fixed`.
- `k`: The number of folds for k-fold cross-validation.
- `val_fraction`: The fraction of the training data to be used for validation.
- `best_epoch_based_on` : measure to base the epoch selection for selecting the optimal model state on. Can be 'val_loss', any of the task default metrics (roc_auc_score, c_index, r2_score), or any of the custom metrics defined.
- `epochs`: The number of epochs for training.
- `batch_size`: The batch size for training.
- `bag_size`: The bag size for MIL models.
- `aggregation_level`: The aggregation level can be `slide` or `patient`. This specifies at which levels bags are aggregated, effectively creating slide-level or patient-level predictions.

### General settings:

- `with_continue`: If `True`, the model will continue training, skipping already finished parameter combinations.
- `task`: The task can be `classification`, `regression`, or `survival`.
- `mode`: The mode can be either `benchmark` or `optimization`.
- `num_workers` : Number of workers for parallelization, set to 0 to disable parallel processing.

# Datasets
The datasets to be used in the project can be specified in the datasets section. One can add any arbitrary number of data sources to a project and specify whether these should be used for training/validation or as testing datasets:
```yaml
datasets:  # List of datasets to be used
  - name: dataset1  # Name of the dataset
    slide_path: path/to/your/slides  # Path to the slide data
    tfrecord_path: path/to/save/tfrecords  # Path to save tfrecords
    tile_path: path/to/save/tiles  # Path to save tiles
    used_for: training  # Whether the dataset is used for training or testing

  - name: dataset2
    slide_path: path/to/your/other/slides
    tfrecord_path: path/to/other/tfrecords
    tile_path: path/to/other/tiles
    used_for: testing
```

# Annotations
The annotation file should be a CSV file with the following columns:

- slide: The name/identifier of the slide, without the file extension (e.g. .svs, .tiff).
- patient: The name/identifier of the patient to which the slide corresponds.
- dataset: The name of the dataset to which the slide belongs.
For classification tasks, the annotation file should also contain a column with the target labels. This column should be named category. For regression tasks, the annotation file should contain a column with the target values. This column should be named value. For survival tasks, the annotation file should contain columns with the survival time and event status. These columns should be named time and event, respectively.

Example of a valid annotation file for a classification task, assuming we use two datasets: dataset1 and dataset2:
```csv
slide,patient,dataset,category
slide1,patient1,dataset1,0
slide2,patient1,dataset1,1
slide3,patient2,dataset1,0
slide4,patient2,dataset1,1
slide5,patient3,dataset1,0
slide6,patient3,dataset1,1
slide7,patient4,dataset2,0
slide8,patient4,dataset2,1
slide9,patient5,dataset2,0
slide10,patient5,dataset2,1
```

For a regression task:
```csv
slide,patient,dataset,value
slide1,patient1,dataset1,0.1
slide2,patient1,dataset1,0.2
slide3,patient2,dataset1,0.3
slide4,patient2,dataset1,0.4
slide5,patient3,dataset1,0.5
slide6,patient3,dataset1,0.6
slide7,patient4,dataset2,0.7
slide8,patient4,dataset2,0.8
slide9,patient5,dataset2,0.9
slide10,patient5,dataset2,1.0
```

For a survival task:
```csv
slide,patient,dataset,time,event
slide1,patient1,dataset1,26,1
slide2,patient1,dataset1,15,1
slide3,patient2,dataset1,16,1
slide4,patient2,dataset1,42,0
slide5,patient3,dataset1,13,1
slide6,patient3,dataset1,11,1
slide7,patient4,dataset2,6,0
slide8,patient4,dataset2,5,1
slide9,patient5,dataset2,84,1
slide10,patient5,dataset2,43,1
```
# Running PathBench
To run pathbench once installed using default setting, one can simply run
```bash
./run_pathbench.sh
```

Note that this script can set your huggingface token for gated models and the configuration file as well as the name of the virtual environment can be changed:
```bash
#If virtual environment does not exist, construct one using pip
if [ ! -d "pathbench_env" ]; then
    python3 -m venv pathbench_env
    source pathbench_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source pathbench_env/bin/activate
fi

#Set slideflow backends
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch

#Set the config file
CONFIG_FILE=conf.yaml

#Run the program
python3 main.py $CONFIG_FILE
```
# Features
- Support for binary/multiclass classification, regression and time-to-event (e.g. survival prediction) problems.
- Benchmarking w.r.t.
    - Tile sizes, magnifications (e.g. 256px, 20x)
    - Normalization methods (e.g. Macenko, Reinhard)
    - Feature extractors (e.g. UNI, GigaPath)
    - MIL aggregators (e.g. CLAM, DSMIL)
    - Loss functions
    - MIL-friendly (feature-space) augmentations
    - Activation functions
- Interpretable visualizations of benchmark output
- Plotly-based benchmark visualization tool
- Efficient Tile processing and QC pipeline inherited by Slideflow
- Optuna-based optimization w.r.t. the benchmark parameters, to quickly find good candidate solutions.

# Package Structure

- pathbench/
  - pathbench/ 
    - benchmarking/
      - benchmark.py # Main benchmarking script
    - experiment/
      - experiment.py # Initialization of experiments
    - models/
      - aggregators.py # MIL aggregation methods
      - feature_extractors.py # Feature extractors
    - utils
      - calculate_feature_similarity.py # Calculate feature extractor similarity
      - utils.py # Util functions
      - losses.py # Houses custom losses for training models
      - augmentations.py # Houses MIL-friendly augmentations
      - metrics.py # Houses custom metrics to calculate during training
    - visualization
      - visualization.py # Houses visualization functions
    - test
      - test.py # Calls testing functions
      - binary_test_conf.yaml # tests binary classification
      - classification_test_conf.yaml # tests multiclass classification
      - opt_test_conf.yaml # tests optimization mode
      - regression_test_conf.yaml # tests regresison
      - survival_test_conf.yaml # tests survival prediciton
  - slideflow_fork # Forked Slideflow package
    - ...
  - requirements.txt
  - README.md
  - LICENSE
  - setup.py # Main setup script for pip
  - setup_pathbench.py # Script to setup a virtual environment and install base packages
  - run_pathbench.sh # Bash script to run pathbench
  - conf.yaml # Default configuration

## Normalization
PathBench currently supports:
- Macenko
- Reinhard
- CycleGan

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
| Virchow2 | Gated | [Link](https://huggingface.co/paige-ai/Virchow2) |
| Hibou-B | Automatic | [Link](https://huggingface.co/histai/hibou-b) |
| Hibou-L | Gated | [Link](https://huggingface.co/histai/hibou-L)
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


## Extending PathBench
PathBench is designed such that it easy to add new feature extractors and MIL aggregation models. 
1. **Custom Feature Extractors**
New feature extractors are added to pathbench/models/feature_extractors.py and follow this format:
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
```
Feature extractor models require a @register_torch descriptor to be recognizable by PathBench as a specifiable feature extractor. Furthermore, the class requires a model to be specified, the embedding size to be specified and a transformation pipeline. For more information, please see the [slideflow documentation](https://slideflow.dev/).
2. **Custom MIL aggregators**
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
3. **Custom Losses**
Custom losses can be added to pathbench/utils/losses.py and need to be specified in the configuration file under custom_loss to be active during benchmarking. As some loss functions rely on attention values to be calculated, one has to specify whether the loss function requires attention, and if so, give it as a parameter to the forward function (attention_weights). Note that PathBench will use the default task loss when an attention-specific loss is specified but the current MIL method does not use/support attention values. An example:
```python
class AttentionEntropyMinimizedCrossEntropyLoss(nn.Module):
    def __init__(self, entropy_lambda: float = 1.0, weight: Optional[Tensor] = None):
        """
        Args:
            entropy_lambda (float): Regularization strength for the entropy minimization.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = True
        self.entropy_lambda = entropy_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(preds, targets)

        # Check if attention weights are normalized
        if attention_weights.dim() > 1:
            attention_weights = torch.softmax(attention_weights, dim=-1)

        # Entropy minimization term
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=1)
        entropy_min_reg = torch.mean(entropy)
        
        # Total loss
        loss = ce_loss + self.entropy_lambda * entropy_min_reg
        return loss
```
4.   **Custom Augmentations**
Custom augmentations are added to augmentations.py and expect the input to be bags of shape (tiles, features). An example:
```python

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
```
6.   **Custom Training Metrics**
Similarly, one can add custom training metrics which will be measured during training. The metrics needs to inheret from fastai's Metric class and have the methods as given down below:
```python
class ConcordanceIndex(Metric):
    """Concordance index metric for survival analysis."""
    def __init__(self):
        self.name = "concordance_index"
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.preds, self.durations, self.events = [], [], []

    def accumulate(self, learn):
        """Accumulate predictions and targets from a batch."""
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        """Accumulate predictions and targets from a batch."""
        preds, targets = to_detach(preds), to_detach(targets)

        # Ensure preds are tensors, handle dict, tuple, and list cases
        if isinstance(preds, dict):
            preds = torch.cat([torch.tensor(v).view(-1) if not isinstance(v, torch.Tensor) else v.view(-1) for v in preds.values()])
        elif isinstance(preds, tuple):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        elif isinstance(preds, list):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        else:
            preds = preds.view(-1) if isinstance(preds, torch.Tensor) else torch.tensor(preds).view(-1)

        # Handle survival targets (durations and events)
        durations = targets[:, 0].view(-1)
        events = targets[:, 1].view(-1)
        
        self.preds.append(preds)
        self.durations.append(durations)
        self.events.append(events)

    @property
    def value(self):
        """Calculate the concordance index."""
        if len(self.preds) == 0: return None
        preds = torch.cat(self.preds).cpu().numpy()
        durations = torch.cat(self.durations).cpu().numpy()
        events = torch.cat(self.events).cpu().numpy()
        ci = concordance_index(durations, preds, events)
        return ci

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
```
## Futher extension
For more fundamental changes, and adding new normalization methods, one needs to change the underlying slideflow code. PathBench uses a [forked version](https://github.com/Sbrussee/slideflow_fork) of the slideflow code, which is added as a submodule in PathBench's repository.

## Developing PathBench
PathBench is currently in its early development stages, but we welcome collaboration on the development of PathBench. Please use pull requests to add new features or open issues if you encounter bugs or would like to see certain features. When encountering bugs, please provide your pathbench configuration along with information on your python environment and OS, this aids in solving the problem in a correct and timely manner.

<p align="center">
  <img src="PathBench-logo-gecentreerd.png" alt="PathBench Logo" width="550" height="400">
</p>
