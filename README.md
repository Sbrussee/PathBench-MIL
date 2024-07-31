<div style="text-align: center;">
 <img src="thumbnail_PathBench-logo-horizontaal" alt="PathBench" width="550" height="400">
</div>

# PathBench
PathBench is a Python package designed to facilitate benchmarking, experimentation, and optimization of Multiple Instance Learning (MIL) for computational histopathology. It provides tools for conducting experiments, evaluating performance, visualizing results, and optimizing hyperparameters. PathBench is built on top of SlideFlow for handling Whole Slide images and integrates Optuna for hyperparameter optimization. PathBench is useful for researchers aiming to benchmarking different pipeline parameters / models for use in their subdomain (in which case the benchmark mode is more suitable) and users starting a Computational Pathology project and wanting to find a suitable pipeline architecture (in which case the optimization mode is more suitable).

PathBench is being developed at the Leiden University Medical Center: Department of Pathology.
Lead Developer: Siemen Brussee

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


<div style="text-align: center;">
 <img src="PathBench-logo-gecentreerd.png" alt="PathBench Logo" width="550" height="400">
</div>
