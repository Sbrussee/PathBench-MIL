Benchmarking Mode
=================

Benchmarking mode in PathBench allows for comprehensive evaluation of different computational pipeline configurations for histopathology tasks. It benchmarks various combinations of normalization methods, feature extractors, and MIL aggregation models.

To configure benchmarking mode, you need to create a configuration file in YAML format. Below is an example configuration file:

.. code-block:: yaml

    experiment:
      project_name: Example_Project
      annotation_file: /path/to/your/annotation_file.csv
      balancing: category # Training set balancing strategy, can be None, category, slide, patient or tile.
      num_workers: 0 # Number of workers for data loading, 0 for no parallelization.
      split_technique: k-fold # Splitting technique, can be k-fold or fixed
      epochs: 5 # Number of training epoch
      best_epoch_based_on: val_loss # Metric to be used for selecting the best training epoch (e.g. val_loss, roc_auc_score, mae, concordance_index)
      batch_size: 32 # Batch size
      bag_size : 512 # Bag size for MIL
      encoder_layers: 1 # Number of encoder layers to use in the MIL aggregator
      z_dim: 256 # Latent space dimensionality in the MIL aggregator
      dropout_p: 0.1 # Dropout probabilitiy in the MIL aggregator
      k: 2 # Number of folds, if split-technique is k-fold
      val_fraction: 0.1 # Fraction of training data to use for validation
      aggregation_level: slide # Aggregation level, can be slide or patient
      task: classification # Task, can be classification, regression or survival

      visualization: 
        - learning_curve
        - confusion_matrix
        - roc_curve

    benchmark:
      parameters:
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

When in the appropriate virtual environment:

.. code-block:: bash

    python3 main.py /path/to/your/config_file.yaml

or using the provided script:

.. code-block:: bash

    ./run_pathbench.sh /path/to/your/config_file.yaml
