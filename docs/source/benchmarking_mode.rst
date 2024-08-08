
Benchmarking Mode
=================

PathBench operates in two modes: Benchmark-mode and Optimization-mode. Benchmark mode takes in different options for the computational pipeline (e.g., normalization methods, feature extractors) and benchmarks all possible combinations, outputting a performance table sorted on mode performance.

To configure benchmarking mode, you need to create a configuration file in YAML format. Below is an example configuration file:

```yaml
experiment:
  project_name: Example_Project
  annotation_file: /path/to/your/annotation_file.csv
  balancing: category
  split_technique: k-fold
  epochs: 5
  batch_size: 32
  bag_size : 512
  k: 2
  val_fraction: 0.1
  aggregation_level: slide
  with_continue: True
  task: classification
  visualization:
    - learning_curve
    - confusion_matrix
    - roc_curve
    - umap
    - mosaic
  mode: benchmark

datasets:
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
    
benchmark_parameters:
  tile_px:
    - 256
  tile_um:
    - 20x
  normalization:
    - macenko
    - reinhard
  feature_extraction:
    - resnet50_imagenet
    - hibou_b
  mil:
    - Attention_MIL
    - dsmil
```

When in the appropriate virtual environment:
```bash
source pathbench_env/bin/activate
```
one can run PathBench in benchmarking mode using the following command:
```bash
python3 main.py /path/to/your/config_file.yaml
```

Or, you can use the provided shell script:
```bash
./run_pathbench.sh /path/to/your/config_file.yaml
```
