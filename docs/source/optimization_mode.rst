Optimization Mode
=================

Optimization mode aims to find the optimal set of computational pipeline hyperparameters to optimize a set performance objective; it will not test all possible combinations.

To configure optimization mode, you need to create a configuration file in YAML format. Below is an example configuration file:

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

      visualization: # Visualization options, options: CLASSIFICATION: confusion_matrix, precision_recall_curve, roc_curve, top_tiles SURVIVAL: survival_roc, concordance_index, calibration REGRESSION: predicted_vs_actual, residuals, qq
        - learning_curve
        - confusion_matrix
        - roc_curve

      evaluation: # Evaluation metrics to use. options: CLASSIFICATION: balanced_accuracy, mean_f1, mean_uncertainty, auc, mean_average_precision, mean_average_recall. REGRESSION: mean_absolute_error, mean_squared_error, r2_score. SURVIVAL: c_index, brier_score.
        - balanced_accuracy
        - auc
      mode: optimization # Mode to use, either benchmark or optimization

    optimization:
      objective_metric: balanced_accuracy # Objective metric to optimize, should also be specified in 'evaluation'
      objective_mode: max # Optimization mode, can be 'max' or 'min'
      objective_dataset: test # Dataset to be used for the objective metric, can be 'val' or 'test'
      sampler: TPESampler # Algorithm to use for optimization e.g. grid_search, TPE, Bayesian
      trials: 100 # Number of optimization trials
      pruner: HyperbandPruner # Pruner for optimization, can be Hyperband, Median etc. Remove this line if you do not want to use a pruner.

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

    benchmark_parameters: # The Benchmark parameters will be used as search space for the optimization
      tile_px: # Tile size in pixels
        - 256

      tile_um: # Tile size (magnification str (e.g 20x, 40x) or microns integer (150, 250))
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

      optimizer: # Optimization algorithm to use for model training (Can be any FastAI optimizer)
        - Adam

When in the appropriate virtual environment:

.. code-block:: bash

    python3 main.py /path/to/your/config_file.yaml

or using the provided script:

.. code-block:: bash

    ./run_pathbench.sh /path/to/your/config_file.yaml