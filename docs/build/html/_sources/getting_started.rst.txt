Getting Started
===============

Setting up a project
---------------------
PathBench inherets the project functionality from SlideFlow. PathBench allows creating projects through the configuration file.
In the configuration, project-specific settings can be specified in the `experiment` section.
The `experiment` section also saves several other important settings:

Project related settings:
- `project_name`: The name of the project.
- `annotation_file`: The path to the annotation file.

Training related settings:
- `balancing`: The balancing technique to be used. This can be 'tile', 'slide', 'patient' or 'category'. Balancing is used to construct training batches with a balanced distribution of classes/patients/slides/tiles.
- `split_technique`: The split technique to be used. This can be 'k-fold' or 'fixed'.
- `k`: The number of folds for k-fold cross-validation.
- `val_fraction`: The fraction of the training data to be used for validation.
- `epochs`: The number of epochs for training.
- `batch_size`: The batch size for training.
- `bag_size`: The bag size for MIL models.
- `aggregation_level`: The aggregation level can be 'slide' or 'patient'. This specifies at which levels bags are aggregated, effectively creating slide-level or patient-level predictions.

General settings:
- `with_continue`: If True, the model will continue training, skipping already finished parameter combinations.
- `task`: The task can be 'classification', 'regression' or 'survival'.
- `mode`: The mode can be either 'benchmark' or 'optimization'.

.. code-block:: yaml

  experiment:
    project_name: MF_TEST # Name of the project
    annotation_file: path/to/your/annotation_file # Path to the annotation file
    balancing: category # Balancing can be 'tile', 'slide', 'patient' or 'category'
    split_technique: k-fold # Split technique can be 'k-fold' or 'fixed'
    k: 3 # Number of folds for k-fold cross-validation
    val_fraction: 0.1 # Fraction of the training data to be used for validation
    epochs: 5 # Number of epochs for training
    batch_size: 32 # Batch size for training
    bag_size: 512 # Bag size for MIL models
    with_continue: True # If True, the model will continue training from the last checkpoint.
    aggregation_level: slide # Aggregation level can be 'slide' or 'patient'
    task: classification # Task can be 'classification', 'regression' or 'survival'
    mode: benchmark # Mode can be either 'benchmark' or 'optimization'.
    #custom_loss:  # Can be any loss as specified in utils/losses.py, or None, in which case the default task loss will be used.
    #custom_metrics: # Can be any list of metrics as specified in utils/metrics.py, or None, in which case the default task metrics will be used.


Datasets
--------
The datasets to be used in the project can be specified in the `datasets` section. One can add any arbitrary number of data sources to a project, and specify whether these should be used for training/validation or as testing datasets:

.. code-block:: yaml

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

Annotations
-----------
As seen in the `experiment` section, the path to the annotation file should be specified. The annotation file should be a CSV file with the following columns:
- `slide`: The name/identifier of the slide.
- `patient`: The name/identifier of the patient to which the slide corresponds.
- `dataset`: The name of the dataset to which the slide belongs.

For classification tasks, the annotation file should also contain a column with the target labels. This column should be named `category`.
For regression tasks, the annotation file should contain a column with the target values. This column should be named `value`.
For survival tasks, the annotation file should contain columns with the survival time and event status. These columns should be named `time` and `event`, respectively.

An example of a valid annotation file for a classification task, assuming we use two datasets: dataset1 and dataset2:

.. code-block:: none

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

An example of a valid annotation file for a regression task:
.. code-block:: none

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

An example of a valid annotation file for a survival task:
.. code-block:: none

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

Note that all the slides we want to use should be present in the annotation file, and the datasets should be specified in the `datasets` section of the configuration file.

Running PathBench
-----------------
To run PathBench once installed using default settings, one can simply run:

.. code-block:: bash

    ./run_pathbench.sh

This script performs the following steps:

1. **If the virtual environment does not exist, construct one using pip:**

.. code-block:: bash

    if [ ! -d "pathbench_env" ]; then
        python3 -m venv pathbench_env
        source pathbench_env/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        source pathbench_env/bin/activate
    fi

2. **Set slideflow backends:**

.. code-block:: bash

    export SF_SLIDE_BACKEND=cucim
    export SF_BACKEND=torch

3. **Set the config file:**

.. code-block:: bash

    CONFIG_FILE=conf.yaml

4. **Run the program:**

.. code-block:: bash

    python3 main.py $CONFIG_FILE