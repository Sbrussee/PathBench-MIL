"""
Benchmarking and Optimization Pipeline
========================================

This module implements a benchmarking and hyperparameter optimization pipeline 
for MIL (Multiple Instance Learning) experiments. It handles dataset splits, 
feature extraction, model training and evaluation, and finally optimization 
using Optuna. The best models’ weights are saved for easy access.

Authors:
    Siemen Brussee, Leiden University Medical Center
"""
import importlib
import os
import sys
import gc
import json
import pickle
import datetime
import logging
import traceback
import random
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import softmax
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, brier_score, concordance_index_ipcw
from sksurv.util import Surv
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, recall_score, average_precision_score, 
    confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, 
    PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import optuna
import optuna.visualization as opt_vis
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import slideflow as sf
from slideflow.model import build_feature_extractor
from slideflow.stats.metrics import ClassifierMetrics
from slideflow.mil import eval_mil, train_mil, mil_config
from slideflow.slide import qc

# Import local modules (adjust the import paths if necessary)
from ..models.feature_extractors import *
from ..models import aggregators
from ..utils.utils import *
from ..visualization.visualization import *
from ..utils.losses import *
from ..utils.metrics import *
from conch.open_clip_custom import create_model_from_pretrained 

# Set logging level for the pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default evaluation metrics for different tasks
DEFAULT_METRICS = {
    'classification': ['balanced_accuracy', 'mean_uncertainty', 'auc', 'mean_average_precision', 'mean_average_recall'],
    'regression': ['mean_absolute_error', 'mean_squared_error', 'r2_score'],
    'survival': ['c_index', 'brier_score'],
    'survival_discrete': ['c_index', 'brier_score']
}

# Essential metrics used for sorting the results (one per task)
ESSENTIAL_METRICS = {
    'classification': ['balanced_accuracy'],
    'regression': ['r2_score'],
    'survival': ['c_index'],
    'survival_discrete': ['c_index']
}

# Built-in MIL methods
BUILT_IN_MIL = ['clam_sb', 'clam_mb', 'attention_mil', 'mil_fc', 'mil_fc_mc', 'transmil', 'bistro.transformer']


def benchmark(config, project):
    """
    Main benchmarking script, which runs the benchmarking based on the configuration

    Args:
        config: The configuration dictionary
        project: The project

    Returns:
        None
    """
    
    logging.info("Starting benchmarking...")
    task = config['experiment']['task']
    benchmark_parameters = config['benchmark_parameters']
    project_directory = f"experiments/{config['experiment']['project_name']}"
    annotations = project.annotations
    logging.info(f"Using {project_directory} for project directory...")
    logging.info(f"Using {annotations} for annotations...")

    #Create a dictionary mapping from the 'patient' column to 'dataset' column in annotations
    annotation_df = pd.read_csv(annotations)
    dataset_mapping = dict(zip(annotation_df['patient'], annotation_df['dataset']))

    #Determine splits file
    splits_file = determine_splits_file(config, project_directory)
    
    #Set the evaluation metrics
    config, evaluation_metrics, aggregation_functions = set_metrics(config)
    logging.info(f"Using {splits_file} splits for benchmarking...")

    #Get all column values
    columns = list(config['benchmark_parameters'].keys())
    columns.extend(list(config['experiment']['evaluation']))                   
    val_df = pd.DataFrame(columns=columns, index=None)
    test_df = pd.DataFrame(columns=columns, index=None)

    all_combinations = calculate_combinations(config)
    logging.info(f"Total number of combinations: {len(all_combinations)}")

    #Create a dictionary to keep track of finished combinations
    finished_combination_dict = {combination: 0 for combination in all_combinations}

    # Iterate over combinations
    for combination in all_combinations:
        combination_dict = {}
        for parameter_name, parameter_value in zip(benchmark_parameters.keys(), combination):
            combination_dict[parameter_name] = parameter_value

        logging.info(f"Running combination: {combination_dict}")

        try:
            target = determine_target_variable(task, config)
            logging.info(f"Target variable: {target}")
            annotation_df = pd.read_csv(project.annotations)

            #Split datasets into train, val and test
            all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                    tile_um=combination_dict['tile_um'],
                                    )
            logging.info(f"Datasets: {all_data}")
            logging.info("Extracting tiles...")

            qc_methods = config['experiment']['qc']
            qc_filters = config['experiment']['qc_filters']

            qc_list = []
            for qc_method in qc_methods:
                #Retrieve the QC method by name from the qc module
                if qc_method == 'Otsu-CLAHE':
                    qc_method = getattr(qc, 'Otsu')(with_clahe=True)
                else:
                    qc_method = getattr(qc, qc_method)()
                qc_list.append(qc_method)

            logging.info(f"QC methods: {qc_list}")
            logging.info(f"QC filter parameters: {qc_filters}")

            #Extract tiles with QC for all datasets
            all_data.extract_tiles(enable_downsample=True,
                                    save_tiles=False,
                                    qc=qc_list,
                                    grayspace_fraction = float(config['experiment']['qc_filters']['grayspace_fraction']),
                                    whitespace_fraction = float(config['experiment']['qc_filters']['whitespace_fraction']),
                                    grayspace_threshold = float(config['experiment']['qc_filters']['grayspace_threshold']),
                                    whitespace_threshold = int(config['experiment']['qc_filters']['whitespace_threshold']),
                                    num_threads = config['experiment']['num_workers'],
                                    report=config['experiment']['report'],)
                                    
            #Select which datasets should be used for training and testing
            datasets = config['datasets']

            # Filter datasets for training
            train_datasets = [d for d in datasets if d['used_for'] == 'training']

            # Filter datasets for testing
            test_datasets = [d for d in datasets if d['used_for'] == 'testing']

            # Assume all_data contains all available datasets
            train_set = all_data.filter(filters={'dataset': [d['name'] for d in train_datasets]})

            # Balance the training dataset
            train_set = balance_dataset(train_set, task, config)

            # Filter test set
            test_set = all_data.filter(filters={'dataset': [d['name'] for d in test_datasets]})
            
            logging.info(f"Train set #slides: {len(train_set.slides())}")
            logging.info(f"Test set #slides: {len(test_set.slides())}")

            logging.info("Splitting datasets...")
            splits = split_datasets(config, project, splits_file, target, project_directory, train_set, dataset_mapping)

            save_string = "_".join([f"{value}" for value in combination_dict.values()])
            string_without_mil = "_".join([f"{value}" for key, value in combination_dict.items() if key != 'mil' and key != 'loss' and key != 'augmentation' and key != 'activation_function' and key != 'optimizer'])
            
            logging.debug(f"Save string: {save_string}") 
            #Run with current parameters
            
            logging.info("Feature extraction...")
            feature_extractor = build_feature_extractor(combination_dict['feature_extraction'].lower(),
                                                        tile_px=combination_dict['tile_px'])
            


            logging.info("Training MIL model...")
            #Generate bags
            bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
            #Set MIL configuration
            mil_conf = set_mil_config(config, combination_dict, task)

            #Create results directory
            os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
            #Create visualization directory
            os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)

            index = 1
            val_results_per_split, test_results_per_split = [], []
            val_pr_per_split, test_pr_per_split = [], []
            survival_results_per_split, test_survival_results_per_split = [], []
            logging.info("Starting training...")
            for train, val in splits:
                logging.info(f"Split {index} started...")
                # Balance the train and val datasets
                train = balance_dataset(train, task, config)
                val = balance_dataset(val, task, config)
                # Train the MIL model
                val_result = project.train_mil(
                    config=mil_conf,
                    outcomes=target,
                    train_dataset=train,
                    val_dataset=val,
                    bags=bags,
                    exp_label=f"{save_string}_{index}",
                    pb_config=config,
                    loss = combination_dict['loss'] if 'loss' in combination_dict else None,
                    augmentation = combination_dict['augmentation'] if 'augmentation' in combination_dict else None,
                    activation_function = combination_dict['activation_function'] if 'activation_function' in combination_dict else None,
                    optimizer = combination_dict['optimizer'] if 'optimizer' in combination_dict else None
                )
                #Get current newest MIL model number
                number = get_highest_numbered_filename(f"experiments/{config['experiment']['project_name']}/mil/")
                #Get the corresponding validation results
                val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
                #Print the unique values in y_pred0:
                print(val_result)
                if config['experiment']['task'] == 'survival' or config['experiment']['task'] == 'survival_discrete':	
                    metrics, durations, events, predictions = calculate_survival_results(val_result)
                    survival_results_per_split.append((durations, events, predictions))
                elif config['experiment']['task'] == 'regression':
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                    y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                    val_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                    val_results_per_split.append([tpr, fpr])
                    val_pr_per_split.extend(val_pr_curves)
                
                val_dict = combination_dict.copy()
                val_dict.update(metrics)

                val_df = val_df.append(val_dict, ignore_index=True)


                #Loop through the test datasets
                for test_dataset in test_datasets:
                    individual_test_set = all_data.filter(filters={'dataset': [test_dataset['name']]})
                    # Define a unique output directory for this test dataset evaluation
                    test_outdir = (
                        f"experiments/{config['experiment']['project_name']}/mil_eval/"
                        f"{number}-{save_string}_{index}_{test_dataset['name']}"
                    )

                    # Evaluate the MIL model on this specific test set
                    _ = eval_mil(
                        weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}",
                        outcomes=target,
                        dataset=individual_test_set,
                        bags=bags,
                        config=mil_conf,
                        outdir=test_outdir,
                        pb_config=config,
                        activation_function=combination_dict.get('activation_function')
                    )

                    # Determine the model string used in the predictions file path
                    if combination_dict['mil'].lower() in BUILT_IN_MIL:
                        model_string = combination_dict['mil'].lower()
                    else:
                        model_string = f"<class 'pathbench.models.aggregators.{combination_dict['mil'].lower()}'>"
                    
                    # Load test predictions for this specific test dataset
                    predictions_path = (
                        f"{test_outdir}/00000-{model_string}/predictions.parquet"
                    )
                    test_result = pd.read_parquet(predictions_path)
                    
                    # Process the test_result based on task type
                    if config['experiment']['task'] in ['survival', 'survival_discrete']:
                        metrics, durations, events, predictions = calculate_survival_results(test_result)
                        test_survival_results_per_split.append((durations, events, predictions))
                    elif config['experiment']['task'] == 'regression':
                        metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                        y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                        test_results_per_split.append((y_true, y_pred))
                    else:
                        metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                        test_results_per_split.append([tpr, fpr])
                        test_pr_per_split.extend(test_pr_curves)
                    
                    # Build a dictionary to record the test results for the current test dataset
                    test_dict = combination_dict.copy()
                    test_dict.update(metrics)
                    test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
                    test_dict['mil_params'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/mil_params.json"
                    test_dict['bag_dir'] = bags
                    test_dict['test_dataset'] = test_dataset['name']
                    
                    # Optionally add the tfrecord directory (depending on your config)
                    if 'x' in str(combination_dict['tile_um']):
                        test_dict['tfrecord_dir'] = f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}"
                    else:
                        test_dict['tfrecord_dir'] = f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}um"
                    
                    # Append this test result to the overall test results DataFrame
                    test_df = test_df.append(test_dict, ignore_index=True)

            # Visualize the top 5 tiles, if applicable
            #Check if model supports attention
            if 'top_tiles' in config['experiment']['visualization']:
                #Select 10 random slides from train, val and test
                train_slides = random.sample(train.slides(), 10)
                val_slides = random.sample(val.slides(), 10)
                test_slides = random.sample(test_set.slides(), 10)


                os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations/top_tiles", exist_ok=True)
                try:
                    output_dir = f"experiments/{config['experiment']['project_name']}/visualizations/top_tiles"
                    for slide_name in val_slides:
                        attention = np.load(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/attention/{slide_name}_att.npz")
                        plot_top5_attended_tiles_per_class(slide_name,
                                                            attention,
                                                            test_dict['tfrecord_dir'],
                                                            output_dir,
                                                            annotation_df, target, "val", str(index), save_string)

                    for slide_name in test_slides:
                        attention = np.load(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}/00000-{model_string}/attention/{slide_name}_att.npz")
                        plot_top5_attended_tiles_per_class(slide_name,
                                                            attention,
                                                            test_dict['tfrecord_dir'],
                                                            output_dir,
                                                            annotation_df, target, "test", str(index), save_string)
                except:
                    logging.warning("Could not plot top 5 attended tiles, attention is likely not available for this model")

            logging.info(f"Combination {save_string} finished...")
            # Save which combinations were finished
            finished_combination_dict[combination] = 1
            with open(f"experiments/{config['experiment']['project_name']}/finished_combinations.pkl", 'wb') as f:
                pickle.dump(finished_combination_dict, f)

            # Save the combination results up to this point, and mark it as finished
            val_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/val_results.csv")
            test_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/test_results.csv")

            plot_across_splits(config, survival_results_per_split, test_survival_results_per_split,
                                val_results_per_split, test_results_per_split, val_pr_per_split, test_pr_per_split,
                                save_string)

            #Close all unused file handles
            gc.collect()
                                    
        except Exception as e:
            logging.warning(f"Combination {save_string} was not succesfully trained due to Error {e}")
            logging.warning(traceback.format_exc())
        

    print(val_df, test_df)
    print(list(benchmark_parameters.keys()))

    val_df_agg, test_df_agg = build_aggregated_results(val_df, test_df, config, benchmark_parameters, aggregation_functions)

    find_and_apply_best_model(config, val_df_agg, test_df_agg, benchmark_parameters, val_df, test_df,
                              val, test_set, target)
    
    # Remove the finished combinations
    with open(f"experiments/{config['experiment']['project_name']}/finished_combinations.pkl", 'wb') as f:
        pickle.dump({}, f)

    # Empty the val and test results
    if os.path.exists(f"experiments/{config['experiment']['project_name']}/results/val_results.csv"):
        os.remove(f"experiments/{config['experiment']['project_name']}/results/val_results.csv")
    if os.path.exists(f"experiments/{config['experiment']['project_name']}/results/test_results.csv"):
        os.remove(f"experiments/{config['experiment']['project_name']}/results/test_results.csv")
    logging.info("Benchmarking finished...")

# =============================================================================
# Dataset and Experiment Utility Functions
# =============================================================================

def determine_splits_file(config: dict, project_directory: str) -> str:
    """
    Determine the splits file name/path based on the experiment configuration.

    Args:
        config (dict): Experiment configuration dictionary.
        project_directory (str): Base directory for the project.

    Returns:
        str: The computed splits file path.
    """
    splits = config['experiment'].get('splits', ".json")
    task = config['experiment']['task']
    if config['experiment']['split_technique'] == 'fixed':
        splits_file = f"{project_directory}/fixed_{task}_{splits}"
    elif config['experiment']['split_technique'] == 'k-fold':
        splits_file = f"{project_directory}/kfold_{task}_{splits}"
    elif config['experiment']['split_technique'] == 'k-fold-stratified':
        splits_file = f"{project_directory}/kfold_stratified_{task}_{splits}"
    else:
        logging.error("Invalid split technique. Please choose either 'fixed' or 'k-fold'.")
        sys.exit(1)
    return splits_file


def set_metrics(config: dict) -> (dict, list, dict):
    """
    Set and adjust evaluation metrics based on the configuration.

    Ensures that essential metrics are always included.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        tuple:
            - Updated configuration (dict)
            - List of evaluation metrics (list)
            - Aggregation functions for each metric (dict)
    """
    if 'evaluation' not in config['experiment'] or not config['experiment']['evaluation']:
        config['experiment']['evaluation'] = DEFAULT_METRICS[config['experiment']['task']]
    else:
        # Ensure essential metrics are included
        essential_metrics = ESSENTIAL_METRICS[config['experiment']['task']]
        config['experiment']['evaluation'] = list(set(config['experiment']['evaluation'] + essential_metrics))
    evaluation_metrics = config['experiment']['evaluation']
    aggregation_functions = {metric: ['mean', 'std'] for metric in evaluation_metrics}
    # Filter only relevant metrics
    aggregation_functions = {metric: agg for metric, agg in aggregation_functions.items() if metric in DEFAULT_METRICS[config['experiment']['task']]}
    return config, evaluation_metrics, aggregation_functions


def calculate_combinations(config: dict) -> list:
    """
    Calculate all parameter combinations based on benchmark_parameters.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        list: All possible combinations of benchmark parameters.
    """
    benchmark_parameters = config['benchmark_parameters']
    combinations = [values for values in benchmark_parameters.values() if isinstance(values, list)]
    all_combinations = list(product(*combinations))
    return all_combinations


def determine_target_variable(task: str, config: dict):
    """
    Determine the target variable (column name) based on the task type.

    Args:
        task (str): The experiment task ('survival', 'classification', 'regression', etc.).
        config (dict): Experiment configuration dictionary.

    Returns:
        list or str: The target variable name(s).

    Raises:
        SystemExit: If an invalid task is specified.
    """
    if task in ['survival', 'survival_discrete']:
        target = ['time', 'event']
    elif task == 'classification':
        target = 'category'
    elif task == 'regression':
        target = 'value'
    else:
        logging.error("Invalid task. Please choose from 'survival', 'survival_discrete', 'classification' or 'regression'.")
        sys.exit(1)
    return target


def balance_dataset(dataset: sf.Dataset, task: str, config: dict) -> sf.Dataset:
    """
    Apply balancing to the dataset based on the task and configuration.

    Args:
        dataset (sf.Dataset): The dataset to be balanced.
        task (str): The task type.
        config (dict): Experiment configuration dictionary.

    Returns:
        sf.Dataset: The balanced dataset (if balancing is specified).
    """
    if 'balancing' not in config['experiment']:
        logging.info("No balancing specified; continuing without balancing.")
        return dataset

    if task in ['survival', 'survival_discrete']:
        headers = ['event']
    elif task == 'classification':
        headers = ['category']
    else:
        logging.info("No balancing needed for regression task.")
        return dataset

    logging.info(f"Balancing dataset on {headers} using '{config['experiment']['balancing']}' strategy.")
    dataset.balance(headers=headers, strategy=config['experiment']['balancing'], force=True)
    return dataset


def split_datasets(config: dict, project: sf.Project, splits_file: str, target: str,
                   project_directory: str, train_set: sf.Dataset, 
                   dataset_mapping: dict = None) -> list:
    """
    Split the dataset into training and validation (or k-fold splits) based on configuration.

    Args:
        config (dict): Experiment configuration dictionary.
        project (sf.Project): The project instance.
        splits_file (str): Path to the splits file.
        target (str): Target variable for splitting.
        project_directory (str): Project directory path.
        train_set (sf.Dataset): The training dataset.
        dataset_mapping (dict): Mapping from patient to dataset (site).

    Returns:
        list: A list of tuples (train, val) for each split.
    """
    if config['experiment']['task'] == 'classification':
        target = "category"
        model_type = 'categorical'
    elif config['experiment']['task'] in ['survival', 'survival_discrete']:
        target = "event"
        model_type = 'categorical'
    elif config['experiment']['task'] == 'regression':
        target = None
        model_type = 'linear'

    if config['experiment']['split_technique'] == 'k-fold' or config['experiment']['split_technique'] == 'k-fold-stratified':
        k = config['experiment']['k']

        if 'stratified' in config['experiment']['split_technique']:
            site_stratified = True
        else:
            site_stratified = False
    
        if os.path.exists(f"{project_directory}/kfold_{splits_file}"):
            splits = train_set.kfold_split(k=k, labels=target, splits=splits_file, read_only=True)
        elif os.path.exists(f"{project_directory}/kfold_stratified_{splits_file}"):
            splits = train_set.kfold_split(k=k, labels=target, splits=splits_file, read_only=True, preserved_site=True)
        else:
            if 'stratified' in config['experiment']['split_technique']:
                splits = train_set.kfold_split(k=k, labels=target, splits=splits_file, preserved_site=True, site_labels=dataset_mapping)
                logging.info(f"Stratified K-fold splits written to {project_directory}/kfold_stratified_{splits}")
            else:
                splits = train_set.kfold_split(k=k, labels=target, splits=splits_file)
                logging.info(f"K-fold splits written to {project_directory}/kfold_{splits}")
    else:
        if os.path.exists(f"{project_directory}/fixed_{splits_file}"):
            train, val = train_set.split(labels=target, model_type=model_type,
                                         val_strategy=config['experiment']['split_technique'],
                                         val_fraction=config['experiment']['val_fraction'],
                                         splits=splits_file, read_only=True)
        else:
            train, val = train_set.split(labels=target, model_type=model_type,
                                         val_strategy=config['experiment']['split_technique'],
                                         val_fraction=config['experiment']['val_fraction'],
                                         splits=splits_file)
            logging.info(f"Fixed splits written to {project_directory}/fixed_{splits_file}")
        splits = [(train, val)]
    return splits


def generate_bags(config: dict, project: sf.Project, all_data: sf.Dataset,
                  combination_dict: dict, string_without_mil: str, feature_extractor) -> str:
    """
    Generate feature bags using a feature extractor and save them to disk.

    Args:
        config (dict): Experiment configuration dictionary.
        project (sf.Project): The project instance.
        all_data (sf.Dataset): The full dataset.
        combination_dict (dict): Dictionary of current combination parameters.
        string_without_mil (str): Identifier string (excluding MIL parameters).
        feature_extractor: The feature extractor model.

    Returns:
        str: The directory path where bags are saved.
    """
    outdir = f"experiments/{config['experiment']['project_name']}/bags"
    os.makedirs(outdir, exist_ok=True)
    bags_dir = f"{outdir}/{string_without_mil}"

    #If some bags are missing, recalculate them
    if "mixed_precision" in config['experiment']:
        bags = project.generate_feature_bags(model=feature_extractor, dataset=all_data,
                                            normalizer=combination_dict['normalization'],
                                            outdir=bags_dir,
                                            mixed_precision=config['experiment']['mixed_precision'])
    else:
        bags = project.generate_feature_bags(model=feature_extractor, dataset=all_data,
                                            normalizer=combination_dict['normalization'],
                                            outdir=bags_dir)
    return bags


def set_mil_config(config: dict, combination_dict: dict, task: str) -> dict:
    """
    Configure the MIL (Multiple Instance Learning) model based on the current combination.

    Args:
        config (dict): Experiment configuration dictionary.
        combination_dict (dict): Dictionary of current combination parameters.
        task (str): The experiment task.

    Returns:
        dict: MIL configuration parameters.
    """
    mil_name = combination_dict['mil'].lower()
    if mil_name not in BUILT_IN_MIL:
        mil_method = get_model_class(aggregators, mil_name)
    else:
        mil_method = mil_name
    mil_conf = mil_config(mil_method,
                          aggregation_level=config['experiment']['aggregation_level'],
                          trainer="fastai",
                          epochs=config['experiment']['epochs'],
                          drop_last=True,
                          batch_size=config['experiment']['batch_size'],
                          bag_size=config['experiment']['bag_size'],
                          task=task)
    return mil_conf


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_across_splits(config: dict, survival_results_per_split: list, test_survival_results_per_split: list,
                       val_results_per_split: list, test_results_per_split: list,
                       val_pr_per_split: list, test_pr_per_split: list, save_string: str) -> None:
    """
    Plot evaluation results across splits for survival, regression, and classification tasks.

    Args:
        config (dict): Experiment configuration dictionary.
        survival_results_per_split (list): Survival results (validation) per split.
        test_survival_results_per_split (list): Survival results (test) per split.
        val_results_per_split (list): Validation results per split.
        test_results_per_split (list): Test results per split.
        val_pr_per_split (list): Precision-recall curves (validation) per split.
        test_pr_per_split (list): Precision-recall curves (test) per split.
        save_string (str): Identifier for saving the plots.

    Returns:
        None
    """
    task = config['experiment']['task']
    if task in ['survival', 'survival_discrete']:
        if 'survival_roc' in config['experiment']['visualization']:
            plot_survival_auc_across_folds(survival_results_per_split, save_string, 'val', config)
            plot_survival_auc_across_folds(test_survival_results_per_split, save_string, 'test', config)
        if 'concordance_index' in config['experiment']['visualization']:
            plot_concordance_index_across_folds(survival_results_per_split, save_string, 'val', config)
            plot_concordance_index_across_folds(test_survival_results_per_split, save_string, 'test', config)
        if 'kaplan_meier' in config['experiment']['visualization']:
            plot_kaplan_meier_curves_across_folds(survival_results_per_split, save_string, 'val', config)
            plot_kaplan_meier_curves_across_folds(test_survival_results_per_split, save_string, 'test', config)
    elif task == 'regression':
        if 'residuals' in config['experiment']['visualization']:
            plot_residuals_across_folds(val_results_per_split, save_string, 'val', config)
            plot_residuals_across_folds(test_results_per_split, save_string, 'test', config)
        if 'predicted_vs_actual' in config['experiment']['visualization']:
            plot_predicted_vs_actual_across_folds(val_results_per_split, save_string, 'val', config)
            plot_predicted_vs_actual_across_folds(test_results_per_split, save_string, 'test', config)
        if 'qq' in config['experiment']['visualization']:
            plot_qq_across_folds(val_results_per_split, save_string, 'val', config)
            plot_qq_across_folds(test_results_per_split, save_string, 'test', config)
    elif task == 'classification':
        if 'roc_curve' in config['experiment']['visualization']:
            plot_roc_curve_across_splits(val_results_per_split, save_string, "val", config)
            plot_roc_curve_across_splits(test_results_per_split, save_string, "test", config)
        if 'precision_recall_curve' in config['experiment']['visualization']:
            plot_precision_recall_across_splits(val_pr_per_split, save_string, "val", config)
            plot_precision_recall_across_splits(test_pr_per_split, save_string, "test", config)


# =============================================================================
# Results Aggregation and Best Model Selection
# =============================================================================

def build_aggregated_results(val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict,
                             benchmark_parameters: dict, aggregation_functions: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Aggregate the results across splits and sort based on the primary evaluation metric.

    Args:
        val_df (pd.DataFrame): Validation results dataframe.
        test_df (pd.DataFrame): Test results dataframe.
        config (dict): Experiment configuration dictionary.
        benchmark_parameters (dict): Dictionary of benchmark parameters.
        aggregation_functions (dict): Dictionary of aggregation functions for each metric.

    Returns:
        tuple: Aggregated (and sorted) validation and test results dataframes.
    """

    #Group by benchmark parameters and 'test_dataset' column in the test dataset
    val_df_grouped = val_df.groupby(list(benchmark_parameters.keys()))
    test_df_grouped = test_df.groupby(list(benchmark_parameters.keys()) + ['test_dataset']) if 'test_dataset' in test_df.columns else list(benchmark_parameters.keys())

    val_df_agg = val_df_grouped.agg(aggregation_functions)
    test_df_agg = test_df_grouped.agg(aggregation_functions)

    val_df_agg.columns = ['_'.join(col).strip() for col in val_df_agg.columns.values]
    test_df_agg.columns = ['_'.join(col).strip() for col in test_df_agg.columns.values]

    # Sort based on the essential metric
    task = config['experiment']['task']
    if task == 'classification':
        val_df_agg = val_df_agg.sort_values(by='balanced_accuracy_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='balanced_accuracy_mean', ascending=False)
    elif task == 'regression':
        val_df_agg = val_df_agg.sort_values(by='r2_score_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='r2_score_mean', ascending=False)
    elif task in ['survival', 'survival_discrete']:
        val_df_agg = val_df_agg.sort_values(by='c_index_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='c_index_mean', ascending=False)

    results_dir = f"experiments/{config['experiment']['project_name']}/results"
    os.makedirs(results_dir, exist_ok=True)
    val_df_agg.to_csv(f"{results_dir}/val_results_agg.csv")
    test_df_agg.to_csv(f"{results_dir}/test_results_agg.csv")
    val_df_agg.to_html(f"{results_dir}/val_results_agg.html")
    test_df_agg.to_html(f"{results_dir}/test_results_agg.html")

    return val_df_agg, test_df_agg



def save_best_model_weights(source_weights_dir: str, config: dict, model_tag: str) -> None:
    """
    Save (copy) the best model weights directory into the dedicated 'saved_models' folder.

    Args:
        source_weights_dir (str): Directory of the source model weights.
        config (dict): Experiment configuration dictionary.
        model_tag (str): A string tag to identify the best model (e.g., 'best_test_model_YYYY-MM-DD_HH-MM-SS').

    Returns:
        None
    """
    dest_dir = f"experiments/{config['experiment']['project_name']}/saved_models/{model_tag}"
    os.makedirs(dest_dir, exist_ok=True)
    # Use system command to copy the directory (or use shutil.copytree for a cross-platform solution)
    os.system(f"cp -r {source_weights_dir} {dest_dir}")
    logging.info(f"Saved best model weights from {source_weights_dir} to {dest_dir}.")


def find_and_apply_best_model(config: dict, val_df_agg: pd.DataFrame, test_df_agg: pd.DataFrame,
                              benchmark_parameters: dict, val_df: pd.DataFrame, test_df: pd.DataFrame,
                              val_dataset: sf.Dataset, test_dataset: sf.Dataset, target: str) -> None:
    """
    Identify the best performing model based on validation and test metrics,
    save its configuration and weights, and then run it on the test set.

    Args:
        config (dict): Experiment configuration dictionary.
        val_df_agg (pd.DataFrame): Aggregated validation results.
        test_df_agg (pd.DataFrame): Aggregated test results.
        benchmark_parameters (dict): Dictionary of benchmark parameters.
        val_df (pd.DataFrame): Detailed validation results.
        test_df (pd.DataFrame): Detailed test results.
        val_dataset (sf.Dataset): Validation dataset.
        test_dataset (sf.Dataset): Test dataset.
        target (str): Target variable.

    Returns:
        None
    """
    task = config['experiment']['task']
    if task == 'classification':
        best_val_model = val_df_agg['balanced_accuracy_mean'].idxmax()
        best_test_model = test_df_agg['balanced_accuracy_mean'].idxmax()
    elif task == 'regression':
        best_val_model = val_df_agg['r2_score_mean'].idxmax()
        best_test_model = test_df_agg['r2_score_mean'].idxmax()
    elif task in ['survival', 'survival_discrete']:
        best_val_model = val_df_agg['c_index_mean'].idxmax()
        best_test_model = test_df_agg['c_index_mean'].idxmax()

    logging.info(f"Best validation model: {best_val_model}")
    logging.info(f"Best test model: {best_test_model}")

    # Create directory for saving the best model
    saved_models_dir = f"experiments/{config['experiment']['project_name']}/saved_models"
    os.makedirs(saved_models_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save best model details as CSV and pickle the model configuration
    best_val_model_path = f"{saved_models_dir}/best_val_model_{now}.csv"
    best_test_model_path = f"{saved_models_dir}/best_test_model_{now}.csv"
    val_df.loc[val_df[benchmark_parameters.keys()].apply(tuple, axis=1) == best_val_model].to_csv(best_val_model_path)
    test_df.loc[test_df[benchmark_parameters.keys()].apply(tuple, axis=1) == best_test_model].to_csv(best_test_model_path)

    # Load best model parameters from the test dataframe
    with open(test_df['mil_params'].iloc[0], 'r') as f:
        best_test_model_dict = json.load(f)
    if 'task' in best_test_model_dict:
        best_test_model_dict['goal'] = best_test_model_dict['task']
        del best_test_model_dict['task']

    if best_test_model_dict['params']['model'].lower() not in BUILT_IN_MIL:
        best_test_model_dict['params']['model'] = getattr(aggregators, best_test_model_dict['params']['model'])
        best_test_model_config = sf.mil.mil_config(trainer=best_test_model_dict['trainer'], **best_test_model_dict['params'])
    else:
        best_test_model_config = None

    best_test_weights = test_df['weights'].iloc[0]
    logging.info(f"Best test weights directory: {best_test_weights}")

    # Run the best model on the test set
    run_best_model(config, 'test', test_dataset, test_df['bag_dir'].iloc[0],
                   best_test_model_config, target, best_test_weights)

    # Save the best model configuration and copy the weights directory for easy access
    best_model_pickle = f"{saved_models_dir}/best_test_model_{now}.pkl"
    with open(best_model_pickle, 'wb') as f:
        pickle.dump(best_test_model_dict, f)
    save_best_model_weights(best_test_weights, config, f"best_test_model_{now}")


def load_class(module_name: str, class_name: str):
    """
    Dynamically load a class from a module.

    Args:
        module_name (str): Name of the module.
        class_name (str): Name of the class.

    Returns:
        class: The loaded class, or None if module_name or class_name is 'None'.
    """
    if module_name == 'None' or class_name == 'None':
        return None
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


# =============================================================================
# Model Training and Evaluation Functions
# =============================================================================

def run_best_model(config: dict, split: int, dataset: sf.Dataset, bags: str,
                   mil_conf: dict, target: str, model_weights: str) -> None:
    """
    Run the best model (using the saved weights) and generate evaluation heatmaps.

    Args:
        config (dict): Experiment configuration dictionary.
        split (int): Split identifier (e.g., 'test').
        dataset (sf.Dataset): Dataset to evaluate.
        bags (str): Directory containing feature bags.
        mil_conf (dict): MIL configuration dictionary.
        target (str): Target variable.
        model_weights (str): Path to the saved model weights.

    Returns:
        None
    """
    logging.info("Running the best model for evaluation...")
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_outdir = f"experiments/{config['experiment']['project_name']}/best_model_eval_{split}_{date}"
    result = eval_mil(
        weights=model_weights,
        outcomes=target,
        dataset=dataset,
        bags=bags,
        config=mil_conf,
        outdir=eval_outdir,
        attention_heatmaps=True,
        cmap="jet",
        norm="linear",
        pb_config=config
    )
    logging.info(f"Best model evaluation completed. Results saved in {eval_outdir}.")


def calculate_results(result: pd.DataFrame, config: dict, save_string: str, dataset_type: str):
    """
    Calculate performance metrics and (for classification) ROC and precision-recall data.

    Args:
        result (pd.DataFrame): Dataframe with predictions and ground truth.
        config (dict): Experiment configuration dictionary.
        save_string (str): Identifier string for saving visualizations.
        dataset_type (str): 'val' or 'test' dataset identifier.

    Returns:
        tuple: (metrics dict, tpr, fpr, precision-recall data)
    """
    metrics = {}
    y_pred_cols = [c for c in result.columns if 'y_pred' in c]
    save_path = f"experiments/{config['experiment']['project_name']}/visualizations"
    os.makedirs(save_path, exist_ok=True)
    task = config['experiment']['task']

    # Rename prediction columns for consistency
    result = result.rename(columns={col: f"y_pred{index}" for index, col in enumerate(y_pred_cols)})
    result = result.rename(columns={col: 'y_true' for col in result.columns if 'y_true' in col})
    y_pred_cols = [f"y_pred{index}" for index in range(len(y_pred_cols))]

    if task == 'classification':
        result['uncertainty'] = result.apply(calculate_entropy, axis=1)
        metrics['mean_uncertainty'] = result['uncertainty'].mean()

    # Initialize lists for classification metrics
    balanced_accuracies, aucs, average_precisions, average_recalls = [], [], [], []
    f1_scores = {}
    precision_recall_data = []
    tpr, fpr = [], []

    if task == 'regression':
        y_true = result['y_true'].values
        y_pred = result['y_pred0'].values
        metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
        metrics['mean_squared_error'] = mean_squared_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        logging.info(f"Regression Metrics - MAE: {metrics['mean_absolute_error']}, MSE: {metrics['mean_squared_error']}, R2: {metrics['r2_score']}")
    else:
        all_y_true = result['y_true'].values
        all_y_pred_prob = []
        unique_classes = np.unique(all_y_true)
        for col in y_pred_cols:
            y_pred_prob = result[col].values
            class_label = int(col[-1])
            all_y_pred_prob.append(y_pred_prob)
            m = ClassifierMetrics(y_true=(all_y_true == class_label).astype(int), y_pred=y_pred_prob)
            fpr_, tpr_, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold

            fpr.append(fpr_)
            tpr.append(tpr_)
            
            optimal_idx = np.argmax(tpr_ - fpr_)
            optimal_threshold = threshold[optimal_idx]
            y_pred_binary_opt = (y_pred_prob > optimal_threshold).astype(int)
            balanced_accuracies.append(balanced_accuracy_score((all_y_true == class_label).astype(int), y_pred_binary_opt))
            f1_scores[col] = f1_score((all_y_true == class_label).astype(int), y_pred_binary_opt)
            aucs.append(auroc)
            average_precisions.append(average_precision_score((all_y_true == class_label).astype(int), y_pred_prob))
            average_recalls.append(recall_score((all_y_true == class_label).astype(int), y_pred_binary_opt))

        all_y_pred_prob = np.vstack(all_y_pred_prob).T
        all_y_pred_class = np.argmax(all_y_pred_prob, axis=1)
        all_cm = confusion_matrix(all_y_true, all_y_pred_class, labels=unique_classes)
        if 'confusion_matrix' in config['experiment']['visualization']:
            disp = ConfusionMatrixDisplay(confusion_matrix=all_cm, display_labels=unique_classes)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Overall Confusion Matrix')
            plt.savefig(f"{save_path}/confusion_matrix_{save_string}_overall.png")
            plt.close()

        y_true_binary = np.isin(all_y_true, unique_classes[unique_classes != unique_classes[-1]]).astype(int)
        y_pred_prob_binary = all_y_pred_prob[:, unique_classes != unique_classes[-1]].max(axis=1)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_prob_binary)
        precision_recall_data.append((precision.flatten(), recall.flatten()))

        metrics['balanced_accuracy'] = np.mean(balanced_accuracies)
        metrics['mean_f1'] = np.mean(list(f1_scores.values()))
        metrics['auc'] = np.mean(aucs)
        metrics['mean_average_precision'] = np.mean(average_precisions)
        metrics['mean_average_recall'] = np.mean(average_recalls)
        logging.info(f"Classification Metrics - Balanced Accuracy: {metrics['balanced_accuracy']}, Mean F1: {metrics['mean_f1']}, AUC: {metrics['auc']}")

    return metrics, tpr, fpr, precision_recall_data


def calculate_survival_results(result: pd.DataFrame):
    """
    Calculate survival metrics (C-index and Brier score) from prediction results.

    Args:
        result (pd.DataFrame): Dataframe containing survival predictions and true outcomes.

    Returns:
        tuple: (metrics dict, durations, events, predictions)
    """
    logging.info("Calculating survival metrics...")
    metrics = {}
    duration_col = [col for col in result.columns if 'duration' in col]
    y_pred_col = [col for col in result.columns if 'y_pred' in col]
    y_true_col = [col for col in result.columns if 'y_true' in col]

    durations = result[duration_col].values
    events = result[y_true_col].values
    predictions = result[y_pred_col].values
    assert durations.shape[0] == events.shape[0] == predictions.shape[0], "Mismatch in number of samples."

    if len(y_pred_col) > 1:
        predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1)[:, np.newaxis]
        predictions = np.sum(predictions * np.arange(predictions.shape[1]), axis=1)

    c_index = concordance_index(durations, predictions, events)
    metrics['c_index'] = c_index
    logging.info(f"Survival C-index: {c_index}")
    brier = calculate_brier_score(durations, events, predictions)
    metrics['brier_score'] = brier
    logging.info(f"Survival Brier Score: {brier}")
    return metrics, durations, events, predictions


def calculate_brier_score(durations: np.array, events: np.array, predictions: np.array) -> float:
    """
    Calculate the Brier score for survival analysis over a range of time points.

    Args:
        durations (np.array): Array of observed durations.
        events (np.array): Array of event indicators.
        predictions (np.array): Array of model predictions.

    Returns:
        float: The average Brier score over the specified time range.
    """
    times = np.linspace(0, durations.max(), 100)
    brier_scores = [np.mean((predictions - (durations <= t) * events) ** 2) for t in times]
    return np.mean(brier_scores)


# =============================================================================
# Optimization with Optuna
# =============================================================================

def optimize_parameters(config: dict, project: sf.Project) -> None:
    """
    Optimize MIL-pipeline parameters using Optuna.

    This function defines an objective function that trains a MIL model with sampled
    hyperparameters and returns a chosen performance metric. After optimization, the
    best parameters are saved and visualizations of the optimization process are generated.
    Additionally, the best trial’s weights are saved in the best model directory.

    Args:
        config (dict): Experiment configuration dictionary.
        project (sf.Project): The slideflow project instance.

    Returns:
        None
    """
    objective_metric = config['optimization']['objective_metric']
    task = config['experiment']['task']
    project_directory = f"experiments/{config['experiment']['project_name']}"

    benchmark_params = config['benchmark_parameters']
    tile_px_choices = benchmark_params.get('tile_px', ['256'])
    tile_um_choices = benchmark_params.get('tile_um', ['20x'])
    normalization_choices = benchmark_params.get('normalization', ['macenko'])
    feature_extraction_choices = benchmark_params.get('feature_extraction', ['uni'])
    mil_choices = benchmark_params.get('mil', ['attention_mil'])
    # For objects (e.g., loss functions), use string identifiers (or factory functions) instead
    loss_choices = benchmark_params.get('loss', ['CrossEntropyLoss'])
    augmentation_choices = benchmark_params.get('augmentation', [None])
    activation_function_choices = benchmark_params.get('activation_function', ['ReLU'])
    optimizer_choices = benchmark_params.get('optimizer', ['Adam'])
    balancing = config['experiment'].get('balancing', None)
    class_weighting = config['experiment'].get('class_weighting', None)

    logging.info("Starting Optuna optimization...")
    logging.info(f"Objective metric: {objective_metric}")
    #Log the search space
    logging.info(f"Search space: {benchmark_params}")

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        This function samples hyperparameters, updates the configuration, and trains a MIL model.
        It logs the current trial number and reports progress. The weights directory for the trial
        is saved as a user attribute for later retrieval if this trial becomes the best trial.

        Args:
            trial (optuna.Trial): The trial object provided by Optuna.

        Returns:
            float: The performance metric (to be maximized or minimized).
        """
        logging.info(f"Starting trial {trial.number}...")
        # Sample hyperparameters
        epochs = trial.suggest_int('epochs', 10, 50)
        batch_size = trial.suggest_int('batch_size', 8, 64)
        z_dim = trial.suggest_int('z_dim', 128, 512)
        encoder_layers = trial.suggest_int('encoder_layers', 1, 3)
        tile_px = trial.suggest_categorical('tile_px', tile_px_choices)
        tile_um = trial.suggest_categorical('tile_um', tile_um_choices)
        normalization = trial.suggest_categorical('normalization', normalization_choices)
        feature_extraction = trial.suggest_categorical('feature_extraction', feature_extraction_choices)
        mil = trial.suggest_categorical('mil', mil_choices)
        loss = trial.suggest_categorical('loss', loss_choices)
        augmentation = trial.suggest_categorical('augmentation', augmentation_choices)
        activation_function = trial.suggest_categorical('activation_function', activation_function_choices)
        optimizer = trial.suggest_categorical('optimizer', optimizer_choices)
        logging.info(f"Trial {trial.number} hyperparameters: epochs={epochs}, batch_size={batch_size}, tile_px={tile_px}, tile_um={tile_um}, normalization={normalization}, feature_extraction={feature_extraction}, mil={mil}")

        # Update configuration with sampled hyperparameters
        config['experiment']['epochs'] = epochs
        config['experiment']['batch_size'] = batch_size
        config['experiment']['z_dim'] = z_dim
        config['experiment']['encoder_layers'] = encoder_layers
        config['experiment']['balancing'] = balancing
        config['experiment']['class_weighting'] = class_weighting

        combination_dict = {
            'tile_px': tile_px,
            'tile_um': tile_um,
            'normalization': normalization,
            'feature_extraction': feature_extraction,
            'mil': mil,
            'loss': loss,
            'augmentation': augmentation,
            'activation_function': activation_function,
            'optimizer': optimizer
        }

        # Initialize dataframes to record results (could be expanded if needed)
        columns = list(config['benchmark_parameters'].keys())
        columns.extend(list(config['experiment']['evaluation']))
        val_df = pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(columns=columns)

        target = determine_target_variable(task, config)
        annotation_df = pd.read_csv(project.annotations)

        # Prepare the dataset
        all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                   tile_um=combination_dict['tile_um'])
        logging.info("Extracting tiles with quality control...")
        qc_methods = config['experiment']['qc']
        qc_filters = config['experiment']['qc_filters']
        qc_list = []
        for qc_method in qc_methods:
            if qc_method == 'Otsu-CLAHE':
                qc_list.append(getattr(qc, 'Otsu')(with_clahe=True))
            else:
                qc_list.append(getattr(qc, qc_method)())
        all_data.extract_tiles(enable_downsample=False, save_tiles=False,
                               qc=qc_list,
                               grayspace_fraction=float(qc_filters['grayspace_fraction']),
                               whitespace_fraction=float(qc_filters['whitespace_fraction']),
                               grayspace_threshold=float(qc_filters['grayspace_threshold']),
                               whitespace_threshold=int(qc_filters['whitespace_threshold']),
                               num_threads=config['experiment']['num_workers'],
                               report=config['experiment']['report'])

        datasets = config['datasets']
        train_datasets = [d for d in datasets if d['used_for'] == 'training']
        test_datasets = [d for d in datasets if d['used_for'] == 'testing']
        train_set = all_data.filter(filters={'dataset': [d['name'] for d in train_datasets]})
        train_set = balance_dataset(train_set, task, config)
        test_set = all_data.filter(filters={'dataset': [d['name'] for d in test_datasets]})

        splits_file = determine_splits_file(config, project_directory)
        splits = split_datasets(config, project, splits_file, target, project_directory, train_set)
        save_string = "_".join([str(value) for value in combination_dict.values()])
        string_without_mil = "_".join([str(value) for key, value in combination_dict.items() if key not in ['mil', 'loss', 'augmentation', 'activation_function']])
        logging.debug(f"Save string: {save_string}")

        logging.info("Starting feature extraction...")
        free_up_gpu_memory()
        feature_extractor = build_feature_extractor(name=combination_dict['feature_extraction'].lower(),
                                                    tile_px=combination_dict['tile_px'])
        logging.info("Generating feature bags...")
        bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
        mil_conf = set_mil_config(config, combination_dict, task)
        os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
        os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)

        index = 1
        val_results_per_split, test_results_per_split = [], []
        val_pr_per_split, test_pr_per_split = [], []
        survival_results_per_split, test_survival_results_per_split = [], []
        logging.info("Starting training for each split...")
        for train, val in splits:
            logging.info(f"Trial {trial.number} - Split {index} started...")
            train = balance_dataset(train, task, config)
            val = balance_dataset(val, task, config)
            # Train the model on the current split
            _ = project.train_mil(
                config=mil_conf,
                outcomes=target,
                train_dataset=train,
                val_dataset=val,
                bags=bags,
                exp_label=f"{save_string}_{index}",
                pb_config=config,
                loss=combination_dict.get('loss'),
                augmentation=combination_dict.get('augmentation'),
                activation_function=combination_dict.get('activation_function'),
                optimizer=combination_dict.get('optimizer')
            )
            number = get_highest_numbered_filename(f"experiments/{config['experiment']['project_name']}/mil/")
            val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
            if task in ['survival', 'survival_discrete']:
                metrics, durations, events, predictions = calculate_survival_results(val_result)
                survival_results_per_split.append((durations, events, predictions))
            elif task == 'regression':
                metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                val_results_per_split.append((y_true, y_pred))
            else:
                metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                val_results_per_split.append([tpr, fpr])
                val_pr_per_split.extend(val_pr_curves)
            val_dict = combination_dict.copy()
            val_dict.update(metrics)
            val_df = val_df.append(val_dict, ignore_index=True)
            # Evaluate the model on the test set
            test_result = eval_mil(
                weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}",
                outcomes=target,
                dataset=test_set,
                bags=bags,
                config=mil_conf,
                outdir=f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}",
                pb_config=config,
                activation_function=combination_dict.get('activation_function')
            )
            if combination_dict['mil'].lower() in BUILT_IN_MIL:
                model_string = combination_dict['mil'].lower()
            else:
                model_string = f"<class 'pathbench.models.aggregators.{combination_dict['mil'].lower()}'>"
            test_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}/00000-{model_string}/predictions.parquet")
            if task in ['survival', 'survival_discrete']:
                metrics, durations, events, predictions = calculate_survival_results(test_result)
                test_survival_results_per_split.append((durations, events, predictions))
            elif task == 'regression':
                metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                test_results_per_split.append((y_true, y_pred))
            else:
                metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                test_results_per_split.append([tpr, fpr])
                test_pr_per_split.extend(test_pr_curves)
            test_dict = combination_dict.copy()
            test_dict.update(metrics)
            test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
            test_dict['mil_params'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/mil_params.json"
            test_dict['bag_dir'] = bags
            test_df = test_df.append(test_dict, ignore_index=True)
            logging.info(f"Trial {trial.number} - Split {index} completed.")
            index += 1

        plot_across_splits(config, survival_results_per_split, test_survival_results_per_split,
                           val_results_per_split, test_results_per_split, val_pr_per_split, test_pr_per_split,
                           save_string)
        # Report progress to the trial
        if config['optimization'].get('pruner'):
            trial.report(np.mean(test_df[objective_metric]), index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Save results (optional) and return the chosen objective metric
        if config['optimization']['objective_dataset'] == 'test':
            measure_df = test_df
        elif config['optimization']['objective_dataset'] == 'val':
            measure_df = val_df
        else:
            raise ValueError("Objective dataset must be 'test' or 'val'.")
        # Save the weights directory for this trial as a user attribute for later retrieval
        trial.set_user_attr("weights_path", test_dict['weights'])
        logging.info(f"Trial {trial.number} completed with {objective_metric}: {np.mean(measure_df[objective_metric])}")
        return np.mean(measure_df[objective_metric])

    # Create optimization directory and study storage
    opt_dir = f"{project_directory}/optimization"
    os.makedirs(opt_dir, exist_ok=True)
    study_name = config['optimization']['study_name']
    #Get a date + time, timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_name = config['optimization']['study_name'] + "_" + timestamp
    storage_path = f"sqlite:///{opt_dir}/optuna_study.db"
    logging.info(f"Using storage at {storage_path} for study '{study_name}'.")
    if config['optimization'].get('load_study') and os.path.exists(storage_path):
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        logging.info(f"Loaded existing study '{study_name}'.")
    else:
        sampler = getattr(optuna.samplers, config['optimization'].get('sampler', 'TPESampler'))()
        pruner = getattr(optuna.pruners, config['optimization'].get('pruner', 'HyperbandPruner'))() if config['optimization'].get('pruner') else None
        direction = 'maximize' if config['optimization']['objective_mode'] == 'max' else 'minimize'
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_path,
            load_if_exists=True if config['optimization'].get('load_study') else False
        )
    logging.info("Starting optimization...")
    study.optimize(objective, n_trials=config['optimization']['trials'])

    # Visualize and save optimization results
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig = opt_vis.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.savefig(f"{opt_dir}/parameter_importances_{current_date}.png")
    plt.close()
    fig = opt_vis.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.savefig(f"{opt_dir}/optimization_history_{current_date}.png")
    plt.close()
    best_params = study.best_params
    with open(f"{opt_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f)
    logging.info(f"Best parameters: {best_params} found in trial {study.best_trial.number} with value: {study.best_value}")

    # Retrieve and save the best model weights from the best trial
    best_weights = study.best_trial.user_attrs.get("weights_path", None)
    if best_weights:
        save_best_model_weights(best_weights, config, f"best_optimized_model_{current_date}")
    logging.info("Optimization finished.")


# =============================================================================
# End of Module
# =============================================================================
