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
| CTransPath | Automatic |  |
| MoCoV3-TransPath | Automatic |  |
| HistoSSL | Automatic | |
| RetCCL | Automatic | |
| PLIP | Automatic | |
| Lunit DINO | Automatic | |
| Lunit SwAV | Automatic | |
| Lunit Barlow Twins | Automatic | |
| Lunit MocoV2 | Automatic | |
| Phikon | Automatic | |
| PathoDuet-HE | Manual | |
| PathoDuet-IHC | Manual | |
| Virchow | Gated | |
| Hibou-B | Automatic | |
| UNI | Gated | |
| Prov-GigaPath | Gated | |
| Kaiko-S8 | Automatic | |
| Kaiko-S16 | Automatic | |
| Kaiko-B8 | Automatic | |
| Kaiko-B16 | Automatic | |
| Kaiko-L14 | Automatic | |
| H-Optimus-0 | Automatic | |

## MIL aggregators
In addition to a wide range of feature extractors, PathBench also includes a wide variety of MIL aggregation methods. Most of these support all tasks (Binary classification, Muliclass classifcation, regression and survival prediction), but some like the CLAM-models only support binary classification. We are actively working on extending support for these models.

| MIL aggregator | Bin. class. | Multi-class. | Regression | Survival | Link |
|----------|----------|----------|----------|----------|----------|
| CLAM-SB | Supported | Not Supported | Not Supported | Not Supported |  |
| CLAM-MB | Supported | Not Supported | Not Supported | Not Supported |  |
| Attention MIL | Supported | Supported | Supported | Supported | |
| TransMIL | Supported | Supported | Supported | Supported | |
| HistoBistro Transformer | Supported | Supported | Supported | Supported | |
| Linear MIL | Supported | Supported | Supported | Supported | |
| Mean MIL | Supported | Supported | Supported | Supported | |
| Max MIL | Supported | Supported | Supported | Supported | |
| Log-Sum-Exp MIL | Supported | Supported | Supported | Supported | |
| LSTM-MIL | Supported | Supported | Supported | Supported | |
| DeepSet-MIL | Supported | Supported | Supported | Supported | |
| Distribution-pool MIL | Supported | Supported | Supported | Supported | |
| VarMIL | Supported | Supported | Supported | Supported | |
| DSMIL | Supported | Supported | Supported | Supported | |


<div style="text-align: center;">
 <img src="PathBench-logo-gecentreerd.png" alt="PathBench Logo" width="550" height="400">
</div>
