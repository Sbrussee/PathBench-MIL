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
import datetime
import sys
import gc
import json
import pickle
import datetime
import logging
import traceback
import random
import shutil
from itertools import product
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import softmax
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, brier_score, concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, recall_score, average_precision_score, 
    confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, 
    PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import optuna
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import slideflow as sf
from slideflow.model import build_feature_extractor
from slideflow.stats.metrics import ClassifierMetrics
from slideflow.mil import eval_mil, train_mil, mil_config
from slideflow.slide import qc

# Import local modules (adjust the import paths if necessary)
from ..models.feature_extractors import *
from ..models import aggregators, slide_level_predictors
from ..utils.utils  import get_available_gpus, remove_cache, get_model_class, get_mil_directory_number, free_up_gpu_memory, calculate_entropy
from ..visualization.visualization import plot_calibration_curve_across_splits, plot_roc_curve_across_splits, \
                            plot_concordance_index_across_folds, plot_kaplan_meier_curves_across_folds, plot_precision_recall_across_splits, \
                            plot_predicted_vs_actual_across_folds, plot_top5_attended_tiles_per_class, plot_survival_auc_across_folds, \
                            plot_residuals_across_folds, plot_qq_across_folds

from conch.open_clip_custom import create_model_from_pretrained 

# Set logging level for the pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Turn off future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

"""
Function that only performs the steps until feature extraction
"""
def extract_features(config : dict, project : sf.Project):
    logging.info("Starting feature extraction...")
    project_directory = f"experiments/{config['experiment']['project_name']}"
    annotations = project.annotations
    logging.info(f"Using {project_directory} for project directory...")
    logging.info(f"Using {annotations} for annotations...")

    annotation_df = pd.read_csv(annotations)
    
    #Caclulate combinations for which to extract features
    all_combinations = calculate_combinations(config)
    logging.info(f"Total number of combinations: {len(all_combinations)}")


    # Iterate over combinations
    for combination in all_combinations:
        logging.info(f"Running combination: {combination}")
        combination_dict = {}
        for parameter_name, parameter_value in zip(config['benchmark_parameters'].keys(), combination):
            combination_dict[parameter_name] = parameter_value
        
        try:

            all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                       tile_um=combination_dict['tile_um'])
            
            logging.info("Extracting tiles...")
            qc_list, qc_filters = build_qc_list(config)
            all_data.extract_tiles(enable_downsample=True,
                                      save_tiles=config['experiment']['save_tiles'] if 'save_tiles' in config['experiment'] else False,
                                      qc=qc_list,
                                      grayspace_fraction = float(qc_filters['grayspace_fraction']),
                                      whitespace_fraction = float(qc_filters['whitespace_fraction']),
                                      grayspace_threshold = float(qc_filters['grayspace_threshold']),
                                      whitespace_threshold = int(qc_filters['whitespace_threshold']),
                                      num_threads = config['experiment']['num_workers'] if 'num_workers' in config['experiment'] else 1,
                                      report=config['experiment']['report'] if 'report' in config['experiment'] else False,
                                      skip_extracted=config['experiment']['skip_extracted'] if 'skip_extracted' in config['experiment'] else True,)
            
            #Set save string
            save_string, string_without_mil = get_save_strings(combination_dict)
            #Define the feature extractor
            feature_extractor = build_feature_extractor(combination_dict['feature_extraction'].lower(),
                                                        tile_px=combination_dict['tile_px'])
            logging.info("Feature extraction...")
            bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
            logging.info(f"Feature extraction for combination {combination} finished...")
        except Exception as e:
            logging.warning(f"Combination {combination} was not succesfully trained due to Error {e}")
            logging.warning(traceback.format_exc())
        
    logging.info("Feature extraction finished...")



def get_column_values(config):
    """
    Get all column values from the benchmark_parameters dictionary that will be used for the
    results dataframes of pathbench.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        list: List of all column values.
    """
    columns = list(config['benchmark_parameters'].keys())
    columns.extend(list(config['experiment']['evaluation']))
    val_df, test_df = pd.DataFrame(columns=columns, index=None), pd.DataFrame(columns=columns, index=None)
    return val_df, test_df

def configure_datasets(config : dict):
    """
    Configure datasets for the experiment based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        list: List of datasets to be used.
    """
    #Select which datasets should be used for training and testing
    datasets = config['datasets'] if 'datasets' in config else [{'name': 'all', 'used_for': 'training'}]

    # Filter datasets for training
    train_datasets = [d for d in datasets if d['used_for'] == 'training']

    # Filter datasets for testing
    test_datasets = [d for d in datasets if d['used_for'] == 'testing']

    if len(train_datasets) == 0:
        logging.error("No training datasets found in the configuration.")
        raise ValueError("No training datasets found in the configuration.")

    if len(test_datasets) == 0:
        logging.warning("No testing datasets found in the configuration. Only getting validation results.")
    
    return train_datasets, test_datasets


def split_train_test(config: dict, all_data: sf.Dataset, task : str):
    """
    Split the dataset into training and testing sets based on the configuration.

    Args:
        config (dict): The configuration dictionary.
        all_data (sf.Dataset): The dataset to be split.
        task (str): The task type.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    train_datasets, test_datasets = configure_datasets(config)
    
    # Assume all_data contains all available datasets
    train_set = all_data.filter(filters={'dataset': [d['name'] for d in train_datasets]})

    # Balance the training dataset
    train_set = balance_dataset(train_set, task, config)

    # Filter test set
    test_set = all_data.filter(filters={'dataset': [d['name'] for d in test_datasets]})
    
    logging.info(f"Train set #slides: {len(train_set.slides())}")
    logging.info(f"Test set #slides: {len(test_set.slides())}")

    return train_set, test_set


def get_save_strings(combination_dict: dict) -> str:
    """
    Generate a save string based on the combination dictionary.

    Args:
        combination_dict (dict): The combination dictionary.

    Returns:
        save_string: The generated save string.
        string_without_mil: The generated string without 'mil', 'loss', 'activation_function', 'optimizer', 'dropout_p', 'encoder_layers', 'z_dim'.
    """
    save_string = "_".join([f"{value}" for key, value in combination_dict.items()])
    
    string_without_mil = "_".join([f"{value}" for key, value in combination_dict.items() if key != 'mil' and key != 'loss' and key != 'augmentation' and key != 'activation_function' and key != 'optimizer' and key != 'dropout_p' and key != 'encoder_layers' and key != 'z_dim' and key != "activation_function"])
    
    logging.debug(f"Save string: {save_string}")
    logging.debug(f"String without MIL: {string_without_mil}")
    return save_string, string_without_mil

def get_model_string(combination_dict: dict, feature_extraction: str, config: dict) -> str:
    """
    Generate a model string based on the combination dictionary and feature extraction method.

    Args:
        combination_dict (dict): The combination dictionary.
        feature_extraction (str): The feature extraction method.
        config (dict): The configuration dictionary.

    Returns:
        str: The generated model string.
    """
    if combination_dict['mil'].lower() in BUILT_IN_MIL:
        model_string = combination_dict['mil'].lower()
    else:
        if "slide" in feature_extraction.lower() and config['experiment']['aggregation_level'] == 'slide':
            model_source_script = "slide_level_predictors"
        else:
            model_source_script = "aggregators"
            
        model_string = f"<class 'pathbench.models.{model_source_script}.{combination_dict['mil'].lower()}'>"
    
    return model_string

def get_default_loss(task: str):
    """
    Get the default loss function based on the task type.

    Args:
        task (str): The task type.

    Returns:
        str: The default loss function.
    """
    if task == 'classification':
        return 'CrossEntropyLoss'
    elif task == 'regression':
        return 'MSELossReg'
    elif task in ['survival']:
        return 'CoxPHLoss'
    elif task in ['survival_discrete']:
        return "NLLLogisticHazardLoss"
    else:
        raise ValueError(f"Unsupported task type: {task}")

def visualize_top5_tiles(config: dict,
                         val: sf.Dataset,
                         test_set: sf.Dataset,
                         annotation_df: pd.DataFrame,
                         target: str,
                         index: int,
                         number: str,
                         save_string: str,
                         model_string: str,
                         test_dict: dict):
    """
    Visualize the top 5 tiles for a given slide based on attention weights.

    Args:
        config (dict): The configuration dictionary.
        val (sf.Dataset): The validation dataset.
        test_set (sf.Dataset): The test dataset.
        annotation_df (pd.DataFrame): The annotation dataframe.
        target (str): The target variable.
        index (int): The index of the current split.
        number (str): The number of the current model.
        save_string (str): The save string for the current combination.
        model_string (str): The model string for the current combination.

    Returns:
        None
    """
    #Select 10 random slides from val and test
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

                
def benchmark(config : dict, project : sf.Project):
    """
    Main benchmarking script, which runs the benchmarking based on the configuration

    Args:
        config: The configuration dictionary
        project: The project

    Returns:
        None
    """
    
    logging.info("Starting benchmarking...")
    task = config['experiment']['task'] if 'task' in config['experiment'] else 'classification'
    benchmark_parameters = config['benchmark_parameters'] if 'benchmark_parameters' in config else None
    project_directory = f"experiments/{config['experiment']['project_name']}" if 'project_name' in config['experiment'] else "experiments/project"
    annotations = project.annotations 

    #Generate an experiment label for the current run
    if 'experiment_label' in config['experiment']:
        experiment_label = config['experiment']['experiment_label']
    else:
        experiment_label = f"benchmarking_{task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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

    val_df, test_df = get_column_values(config)

    all_combinations = calculate_combinations(config)
    logging.info(f"Total number of combinations: {len(all_combinations)}")

    # Iterate over combinations
    combinations_successfully_finished = 0
    for combination in all_combinations:
        combination_dict = {}
        for parameter_name, parameter_value in zip(benchmark_parameters.keys(), combination):
            combination_dict[parameter_name] = parameter_value

        logging.info(f"Running combination: {combination_dict}")

        save_string, string_without_mil = get_save_strings(combination_dict)

        try:
            target = determine_target_variable(task, config)
            logging.info(f"Target variable: {target}")

            #Split datasets into train, val and test
            all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                    tile_um=combination_dict['tile_um'],
                                    )
            logging.info(f"Datasets: {all_data}")
            logging.info("Extracting tiles...")

            qc_list, qc_filters = build_qc_list(config)

            logging.info(f"QC methods: {qc_list}")

            # NOTE(pvalkema): if num_workers is 0 or 1 and this value is used for num_threads, an error occurs during tile extraction.
            # TODO: We need to either fix the bug in extract_tiles or decide to keep the workaround below
            tile_extraction_num_threads = config['experiment']['num_workers'] if 'num_workers' in config['experiment'] else 4
            if tile_extraction_num_threads <= 1:
                tile_extraction_num_threads = 4

            #Extract tiles with QC for all datasets
            try:
                all_data.extract_tiles(enable_downsample=True,
                                        save_tiles=config['experiment']['save_tiles'] if 'save_tiles' in config['experiment'] else False,
                                        qc=qc_list,
                                        grayspace_fraction = float(qc_filters['grayspace_fraction']),
                                        whitespace_fraction = float(qc_filters['whitespace_fraction']),
                                        grayspace_threshold = float(qc_filters['grayspace_threshold']),
                                        whitespace_threshold = int(qc_filters['whitespace_threshold']),
                                        num_threads = tile_extraction_num_threads,
                                        report=config['experiment']['report'] if 'report' in config['experiment'] else False,
                                        skip_extracted=config['experiment']['skip_extracted'] if 'skip_extracted' in config['experiment'] else True,
                )
            except Exception as e:
                logging.error(f"tile extraction failed: {e}")
                traceback.print_exc()
                sys.exit()

            train_datasets, test_datasets = configure_datasets(config)

            train_set, test_set = split_train_test(config, all_data, task)

            logging.info("Splitting datasets...")
            splits = split_datasets(config, project, splits_file, target, project_directory, train_set, dataset_mapping)

            #Run with current parameters
            
            logging.info("Feature extraction...")
            feature_extractor = build_feature_extractor(combination_dict['feature_extraction'].lower(),
                                                        tile_px=combination_dict['tile_px'])

            logging.info("Training MIL model...")

            slide_level = True if "slide" in combination_dict['feature_extraction'].lower() else False
            #Generate bags
            bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
            #Set MIL configuration
            mil_conf, combination_dict = set_mil_config(config, combination_dict, task, slide_level)

            #Create results directory
            os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
            #Create visualization directory
            os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)

            index = 1
            #Create lists to store validation results
            val_results_per_split = []
            val_pr_per_split = []
            survival_results_per_split = []
            #Create test results lists per split, per dataset
            test_results_per_split = { ds['name'] : [] for ds in test_datasets }
            test_pr_per_split = { ds['name'] : [] for ds in test_datasets }
            test_survival_results_per_split = { ds['name'] : [] for ds in test_datasets }
            logging.info("Starting training...")
            for train, val in splits:
                logging.info(f"Split {index} started...")
                # Balance the train and val datasets
                train = balance_dataset(train, task, config)
                val = balance_dataset(val, task, config)
                
                if task in ['survival', 'survival_discrete']:
                    dur_tr, evt_tr = _surv_labels_from_dataset(train)



                model_kwargs = {
                    'pb_config' : config,
                    'loss' : combination_dict['loss'] if 'loss' in combination_dict else get_default_loss(task),
                }
                logging.debug(f"Model kwargs before passing to slideflow: {model_kwargs}")

                # Train the MIL model
                val_result = project.train_mil(
                    config=mil_conf,
                    outcomes=target,
                    train_dataset=train,
                    val_dataset=val,
                    bags=bags,
                    exp_label=f"{save_string}_{index}",
                    attention_heatmaps=config['experiment']['save_heatmaps_val'] if 'attention_heatmaps_val' in config['experiment'] else False,
                    **model_kwargs)
                
                mil_directory = f"experiments/{config['experiment']['project_name']}/mil"

                number = get_mil_directory_number(mil_directory, save_string)

                def get_validation_metrics(number: str):
                    #Get current newest MIL model number
                    number = get_mil_directory_number(mil_directory, save_string)
                    #Get the corresponding validation results
                    val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
                    val_result.to_csv(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.csv", index=False)
                    logging.info(f"Validation results saved to experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
                    if task == 'survival' or task == 'survival_discrete':	
                        metrics, durations, events, predictions = calculate_survival_results(val_result, invert_preds=False, y_train=(dur_tr, evt_tr))
                        survival_results_per_split.append((durations, events, predictions))
                    elif task  == 'regression':
                        metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                        y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                        val_results_per_split.append((y_true, y_pred))
                    else:
                        metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                        val_results_per_split.append([tpr, fpr])
                        val_pr_per_split.extend(val_pr_curves)
                    
                    return metrics

                metrics = get_validation_metrics(number)
                val_dict = combination_dict.copy()
                val_dict.update(metrics)
                val_df = val_df.append(val_dict, ignore_index=True)

                #Loop through the test datasets
                if test_datasets:
                    for test_dataset in test_datasets:
                        logging.info(f"Evaluating on test dataset: {test_dataset['name']}")
                        #Only select the test set at hand
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
                            **model_kwargs
                        )

                        model_string = get_model_string(combination_dict, combination_dict['feature_extraction'], config = config)

                        def get_test_metrics(model_string: str, test_outdir: str):
                            # Load the test predictions for this specific test dataset
                            predictions_path = (
                                f"{test_outdir}/00000-{model_string}/predictions.parquet"
                            )
                            test_result = pd.read_parquet(predictions_path)
                            test_result.to_csv(f"{test_outdir}/00000-{model_string}/predictions.csv", index=False)
                            # Process the test_result based on task type
                            if task in ['survival', 'survival_discrete']:
                                metrics, durations, events, predictions = calculate_survival_results(test_result, invert_preds=False, y_train=(dur_tr, evt_tr))
                                test_survival_results_per_split[test_dataset['name']].append((durations, events, predictions))
                            elif task == 'regression':
                                metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                                y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                                test_results_per_split[test_dataset['name']].append((y_true, y_pred))
                            else:
                                metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                                test_results_per_split[test_dataset['name']].append([tpr, fpr])
                                test_pr_per_split[test_dataset['name']].extend(test_pr_curves)
        
                            return metrics
                        
                        metrics = get_test_metrics(model_string, test_outdir)
                        # Build a dictionary to record the test results for the current test dataset
                        test_dict = combination_dict.copy()
                        test_dict.update(metrics)
                        test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
                        test_dict['mil_params'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/mil_params.json"
                        test_dict['bag_dir'] = bags
                        test_dict['test_dataset'] = test_dataset['name']
                        
                        logging.debug(f"Test weights: {test_dict['weights']}")
                        logging.debug(f"Test MIL params: {test_dict['mil_params']}")
                        logging.debug(f"Test bag directory: {test_dict['bag_dir']}")
                        logging.debug(f"Test dataset: {test_dict['test_dataset']}")

                        # Optionally add the tfrecord directory (depending on your config)
                        if 'x' in str(combination_dict['tile_um']):
                            test_dict['tfrecord_dir'] = f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}"
                        else:
                            test_dict['tfrecord_dir'] = f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}um"
                        
                        # Append this test result to the overall test results DataFrame
                        test_df = test_df.append(test_dict, ignore_index=True)
            
                else:
                    logging.info("No test datasets found in the configuration. Skipping test evaluation.")

                # Increment the index for the next split
                index += 1

            



            # Visualize the top 5 tiles, if applicable
            #Check if model supports attention
            if 'top_tiles' in config['experiment']['visualization']:
                visualize_top5_tiles(config, val, test_set, annotation_df, target, index, number, save_string, model_string, test_dict)

            logging.info(f"Combination {save_string} finished...")
            combinations_successfully_finished += 1

            # Save the combination results up to this point, and mark it as finished
            val_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/val_results_{experiment_label}.csv")
            test_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/test_results_{experiment_label}.csv")

            plot_across_splits(config, survival_results_per_split, test_survival_results_per_split,
                        val_results_per_split, test_results_per_split, val_pr_per_split, test_pr_per_split,
                                save_string, invert_preds=False)

            #Close all unused file handles
            gc.collect()
                                    
        except Exception as e:
            logging.warning(f"Combination {save_string} was not succesfully trained due to Error {e}")
            logging.warning(traceback.format_exc())
        

    logging.info(f"Combinations successfully finished: {combinations_successfully_finished}")
    if combinations_successfully_finished > 0:
        print(val_df, test_df)
        print(list(benchmark_parameters.keys()))

        plot_benchmarking_output(config, val_df, test_df, experiment_label)

        val_df_agg, test_df_agg = build_aggregated_results(val_df, test_df, config, benchmark_parameters,
                                                           aggregation_functions, experiment_label)

        find_and_apply_best_model(config, val_df_agg, test_df_agg, benchmark_parameters, val_df, test_df,
                                  val, test_set, target, slide_level)

    # Empty the val and test results
    if os.path.exists(f"experiments/{config['experiment']['project_name']}/results/val_results_{experiment_label}.csv"):
        os.remove(f"experiments/{config['experiment']['project_name']}/results/val_results_{experiment_label}.csv")
    if os.path.exists(f"experiments/{config['experiment']['project_name']}/results/test_results_{experiment_label}.csv"):
        os.remove(f"experiments/{config['experiment']['project_name']}/results/test_results_{experiment_label}.csv")
    logging.info("Benchmarking finished...")


def plot_benchmarking_output(config: dict, val_df: pd.DataFrame, test_df: pd.DataFrame, experiment_label: str):
    """
    Plot boxplots of each evaluation metric for each parameter combination using
    the unaggregated val_df and test_df, with shortened labels, dynamic sizing,
    sorted by mean, and unique colors.
    """
    # Create a timestamp string for filenames
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define output directory for the plots and create it if necessary
    output_dir = f"experiments/{config['experiment']['project_name']}/visualizations/benchmarking"
    os.makedirs(output_dir, exist_ok=True)

    # Identify all parameter keys
    param_keys = list(config['benchmark_parameters'].keys())

    # Determine which parameters vary across all runs (val + test)
    combined = pd.concat([val_df[param_keys], test_df[param_keys]], ignore_index=True)
    varying_params = [k for k in param_keys if combined[k].nunique() > 1]
    if not varying_params:
        varying_params = param_keys

    # Function to build short labels from varying params
    def make_labels(df):
        if df.empty:
            return pd.Series([], dtype=str, index=df.index)
        return df[varying_params].astype(str).agg('_'.join, axis=1)

    # Get evaluation metrics
    metrics = config['experiment'].get('evaluation', [])

    # --- Validation Plots ---
    val_df_proc = val_df.copy()
    val_df_proc['combination'] = make_labels(val_df_proc)

    for metric in metrics:
        if metric not in val_df_proc.columns:
            continue

        # Prepare grouped data
        grouped = val_df_proc.groupby('combination')[metric].apply(list)
        labels = grouped.index.tolist()
        data = grouped.tolist()

        # Sort by mean
        means = [np.mean(d) for d in data]
        order = np.argsort(means)
        sorted_data = [data[i] for i in order]
        sorted_labels = [labels[i] for i in order]

        # Dynamic figure size based on number of boxes
        fig_width = max(10, len(sorted_labels) * 0.5)
        plt.figure(figsize=(fig_width, 6))

        # Create boxplot with unique colors
        box = plt.boxplot(sorted_data, labels=sorted_labels, patch_artist=True, showfliers=True)
        cmap = plt.get_cmap('tab20')
        colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(len(sorted_labels))]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3)
        plt.title(f"Validation {metric} by Parameter Combination")
        plt.xlabel("Parameter Combination")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/val_{metric}_boxplot_{experiment_label}.png")
        plt.close()

    # --- Test Plots ---
    test_df_proc = test_df.copy()
    test_df_proc['combination'] = make_labels(test_df_proc)

    for ds in test_df_proc['test_dataset'].unique():
        subset = test_df_proc[test_df_proc['test_dataset'] == ds]
        for metric in metrics:
            if metric not in subset.columns:
                continue

            grouped = subset.groupby('combination')[metric].apply(list)
            labels = grouped.index.tolist()
            data = grouped.tolist()

            means = [np.mean(d) for d in data]
            order = np.argsort(means)
            sorted_data = [data[i] for i in order]
            sorted_labels = [labels[i] for i in order]

            fig_width = max(10, len(sorted_labels) * 0.5)
            plt.figure(figsize=(fig_width, 6))

            box = plt.boxplot(sorted_data, labels=sorted_labels, patch_artist=True, showfliers=True)
            cmap = plt.get_cmap('tab20')
            colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(len(sorted_labels))]
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.3)
            plt.title(f"Test ({ds}) {metric} by Parameter Combination")
            plt.xlabel("Parameter Combination")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/test_{ds}_{metric}_boxplot_{experiment_label}.png")
            plt.close()


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
    #Splits can be given as a hyperparameter, otherwise defaults to {method}_{task}.json
    splits = config['experiment'].get('splits', ".json")
    task = config['experiment'].get('task', 'classification')
    if config['experiment']['split_technique'] == 'fixed':
        splits_file = f"{project_directory}/fixed_{task}{splits}"
    elif config['experiment']['split_technique'] == 'k-fold':
        splits_file = f"{project_directory}/kfold_{task}{splits}"
    elif config['experiment']['split_technique'] == 'k-fold-stratified':
        splits_file = f"{project_directory}/kfold_stratified_{task}{splits}"
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
    dataset_with_balancing_info = dataset.balance(headers=headers, strategy=config['experiment']['balancing'], force=True)
    return dataset_with_balancing_info


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
    #Split by category when task is classification
    if config['experiment']['task'] == 'classification':
        target = "category"
        model_type = 'categorical'
    #Split by event when task is survival
    elif config['experiment']['task'] in ['survival', 'survival_discrete']:
        target = "event"
        model_type = 'categorical'
    #Do not split by category when task is regression
    elif config['experiment']['task'] == 'regression':
        target = None
        model_type = 'linear'


    logging.info(f"Splitting dataset using {config['experiment']['split_technique']} with target: {target}")
    logging.debug(f"Splits file: {splits_file}")
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
                logging.info(f"Stratified K-fold splits written to {splits_file}")
            else:
                splits = train_set.kfold_split(k=k, labels=target, splits=splits_file)
                logging.info(f"K-fold splits written to {splits_file}")
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

    logging.info(f"Creating bags in {bags_dir}...")

    #Determine how many GPUs are available
    num_gpus = get_available_gpus()
    logging.info(f"Number of GPUs available for feature extraction: {num_gpus}")

    #Check if on server environment
    if "SLURM_JOB_ID" in os.environ:
        logging.info("Running on a SLURM server environment, removing torch/huggingface cache to avoid full cache errors.")
        remove_cache()

    if combination_dict['normalization'].lower() == "none":
        combination_dict['normalization'] = None

    #If some bags are missing, recalculate them
    if "mixed_precision" in config['experiment']:
        bags = project.generate_feature_bags(model=feature_extractor, dataset=all_data,
                                            normalizer=combination_dict['normalization'],
                                            outdir=bags_dir,
                                            mixed_precision=config['experiment']['mixed_precision'],
                                            num_gpus=num_gpus,
                                            force_regenerate=not config['experiment']['skip_feature_extraction'] if 'skip_feature_extraction' in config['experiment'] else False,
                                            progress=False,
                                            num_workers=config['experiment']['num_workers'] if 'num_workers' in config['experiment'] else 4
        )
    else:
        bags = project.generate_feature_bags(model=feature_extractor, dataset=all_data,
                                            normalizer=combination_dict['normalization'],
                                            outdir=bags_dir,
                                            num_gpus=num_gpus,
                                            force_regenerate=not config['experiment']['skip_feature_extraction'] if 'skip_feature_extraction' in config['experiment'] else False,
                                            progress=False,
                                            num_workers=config['experiment']['num_workers'] if 'num_workers' in config['experiment'] else 4)
    return bags


def build_qc_list(config: dict):
    """
    Build a list of quality control methods based on the configuration.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        list: List of quality control methods.
    """
    #Set QC methods, if not present, set to None
    qc_methods = config['experiment']['qc'] if 'qc' in config['experiment'] else []
    #Set QC filters, if not present, set to slideflow defaults
    qc_filters = config['experiment']['qc_filters'] if 'qc_filters' in config['experiment'] else {'whitespace_fraction' : 1,
                                                                                                  'whitespace_threshold' : 230,
                                                                                                  'grayspace_fraction' : 1,
                                                                                                  'grayspace_threshold' : 0.01}
    logging.debug(f"QC filters set: {qc_filters}")
    #Skip QC if no methods are specified
    qc_list = []
    if qc_methods is None:
        logging.warning("No QC methods specified, not applying any QC algorithms.")
        return None, qc_filters

    logging.debug(f"QC methods set: {qc_methods}")
    for qc_method in qc_methods:
        #Retrieve the QC method by name from the qc module
        if qc_method == 'Otsu-CLAHE':
            qc_method = getattr(qc, 'Otsu')(with_clahe=True)
        else:
            qc_method = getattr(qc, qc_method)()
        qc_list.append(qc_method)

    return qc_list, qc_filters

def set_mil_config(config: dict, combination_dict: dict, task: str, slide_level: bool = False) -> dict:
    """
    Configure the MIL (Multiple Instance Learning) model based on the current combination.

    Args:
        config (dict): Experiment configuration dictionary.
        combination_dict (dict): Dictionary of current combination parameters.
        task (str): The experiment task.

    Returns:
        dict: MIL configuration parameters.
    """
    # Check for fallback to default MIL method
    if 'mil' not in combination_dict or not combination_dict['mil']:
        logging.warning("No MIL method specified. Using default attention MIL method.")
        #Setting default attention_mil
        mil_name = 'attention_mil'
        combination_dict['mil'] = mil_name
    else:
        mil_name = combination_dict['mil'].lower()

    # Check if the MIL method is slideflow built-in or external
    if mil_name not in BUILT_IN_MIL:
        #check if correct MIL methods chosen for slide-level. If not, will fall back to MLP classifier.
        if slide_level and config['experiment']['aggregation_level'] == 'slide':
            try:
                mil_method = get_model_class(slide_level_predictors, mil_name)
            except:
                logging.warning(f"You are either using an undefined slide predictor head or an MIL model: {mil_name}, Now falling back to Slide-level MLP classifier")
                mil_method = get_model_class(slide_level_predictors, 'mlp_slide_classifier')
                combination_dict['mil'] = 'mlp_slide_classifier'
                pass
        else:
            #If not slide-level, use the MIL method as specified
            mil_method = get_model_class(aggregators, mil_name)
    else:
        if slide_level and config['experiment']['aggregation_level'] == 'slide':
            logging.warning("You are using a bag-level model for slide-level predictions. Please use a slide-level model.")
            mil_method = get_model_class(slide_level_predictors, 'mlp_slide_classifier')
            combination_dict['mil'] = 'mlp_slide_classifier'
        else:
            mil_method = mil_name

    if slide_level:
        mil_conf = mil_config(mil_method,
                            aggregation_level=config['experiment']['aggregation_level'] if 'aggregation_level' in config['experiment'] else 'slide',
                            trainer="lightning",
                            epochs=config['experiment']['epochs'] if 'epochs' in config['experiment'] else 50,
                            batch_size=config['experiment']['batch_size'] if 'batch_size' in config['experiment'] else 64,
                            bag_size=None,
                            z_dim = config['experiment']['z_dim'] if 'z_dim' in config['experiment'] else 512,
                            encoder_layers = config['experiment']['encoder_layers'] if 'encoder_layers' in config['experiment'] else 1,
                            dropout_p = config['experiment']['dropout_p'] if 'dropout_p' in config['experiment'] else 0.25,
                            activation_function = combination_dict['activation_function'] if 'activation_function' in combination_dict else 'ReLU',
                            slide_level=True,
                            task=task)
    else:
        mil_conf = mil_config(mil_method,
                            aggregation_level=config['experiment']['aggregation_level'] if 'aggregation_level' in config['experiment'] else 'slide',   
                            trainer="lightning",
                            epochs=config['experiment']['epochs'] if 'epochs' in config['experiment'] else 50,
                            drop_last=True,
                            batch_size=config['experiment']['batch_size'] if 'batch_size' in config['experiment'] else 64,
                            bag_size=config['experiment']['bag_size'] if 'bag_size' in config['experiment'] else 512,
                            z_dim = config['experiment']['z_dim'] if 'z_dim' in config['experiment'] else 512,
                            encoder_layers = config['experiment']['encoder_layers'] if 'encoder_layers' in config['experiment'] else 1,
                            dropout_p = config['experiment']['dropout_p'] if 'dropout_p' in config['experiment'] else 0.25,
                            activation_function = combination_dict['activation_function'] if 'activation_function' in combination_dict else 'ReLU',
                            slide_level=False,
                            task=task)
    return mil_conf, combination_dict


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_across_splits(config: dict, survival_results_per_split: list, test_survival_results_per_split: Dict[str, list],
                       val_results_per_split: list, test_results_per_split: Dict[str, list],
                       val_pr_per_split: list, test_pr_per_split: Dict[str, list], save_string: str,
                       invert_preds: bool) -> None:
    """
    Plot evaluation results across splits for survival, regression, and classification tasks.

    Args:
        config (dict): Experiment configuration dictionary.
        survival_results_per_split (list): Survival results (validation) per split.
        test_survival_results_per_split (dict): Survival results (test) per split, per dataset.
        val_results_per_split (list): Validation results per split.
        test_results_per_split (dict): Test results per split, per dataset.
        val_pr_per_split (list): Precision-recall curves (validation) per split.
        test_pr_per_split (dict): Precision-recall curves (test) per split, per dataset.
        save_string (str): Identifier for saving the plots.

    Returns:
        None
    """
    task = config['experiment']['task']
    if task in ['survival', 'survival_discrete']:
        if 'survival_roc' in config['experiment']['visualization']:
            plot_survival_auc_across_folds(survival_results_per_split, save_string, 'val', config, invert_preds=invert_preds)
        if 'concordance_index' in config['experiment']['visualization']:
            plot_concordance_index_across_folds(survival_results_per_split, save_string, 'val', config)
        if 'kaplan_meier' in config['experiment']['visualization']:
            plot_kaplan_meier_curves_across_folds(survival_results_per_split, save_string, 'val', config, invert_preds=invert_preds)
    elif task == 'regression':
        if 'residuals' in config['experiment']['visualization']:
            plot_residuals_across_folds(val_results_per_split, save_string, 'val', config)
        if 'predicted_vs_actual' in config['experiment']['visualization']:
            plot_predicted_vs_actual_across_folds(val_results_per_split, save_string, 'val', config)
        if 'qq' in config['experiment']['visualization']:
            plot_qq_across_folds(val_results_per_split, save_string, 'val', config)
    elif task == 'classification':
        if 'roc_curve' in config['experiment']['visualization']:
            plot_roc_curve_across_splits(val_results_per_split, save_string, "val", config)
        if 'precision_recall_curve' in config['experiment']['visualization']:
            plot_precision_recall_across_splits(val_pr_per_split, save_string, "val", config)

    if test_results_per_split != {}:
        for ds_name, splits in test_results_per_split.items():
            if task == 'classification':
                if 'roc_curve' in config['experiment']['visualization']:
                    plot_roc_curve_across_splits(splits, save_string, f"test_{ds_name}", config)
                if 'precision_recall_curve' in config['experiment']['visualization']:
                    plot_precision_recall_across_splits(test_pr_per_split[ds_name], save_string, f"test_{ds_name}", config)
            elif task == 'regression':
                if 'residuals' in config['experiment']['visualization']:
                    plot_residuals_across_folds(splits, save_string, f"test_{ds_name}", config)
                if 'predicted_vs_actual' in config['experiment']['visualization']:
                    plot_predicted_vs_actual_across_folds(splits, save_string, f"test_{ds_name}", config)
                if 'qq' in config['experiment']['visualization']:
                    plot_qq_across_folds(splits, save_string, f"test_{ds_name}", config)
            elif task in ['survival', 'survival_discrete']:
                if 'survival_roc' in config['experiment']['visualization']:
                    plot_survival_auc_across_folds(test_survival_results_per_split[ds_name], save_string, f"test_{ds_name}", config, invert_preds=invert_preds)
                if 'concordance_index' in config['experiment']['visualization']:
                    plot_concordance_index_across_folds(test_survival_results_per_split[ds_name], save_string, f"test_{ds_name}", config)
                if 'kaplan_meier' in config['experiment']['visualization']:
                    plot_kaplan_meier_curves_across_folds(test_survival_results_per_split[ds_name], save_string, f"test_{ds_name}", config, invert_preds=invert_preds)
        logging.info(f"Plots saved to experiments/{config['experiment']['project_name']}/visualizations/benchmarking/{save_string}.png")


# =============================================================================
# Results Aggregation and Best Model Selection
# =============================================================================

def build_aggregated_results(val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict,
                             benchmark_parameters: dict, aggregation_functions: dict,
                             experiment_label: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Aggregate the results across splits and sort based on the primary evaluation metric.

    Args:
        val_df (pd.DataFrame): Validation results dataframe.
        test_df (pd.DataFrame): Test results dataframe.
        config (dict): Experiment configuration dictionary.
        benchmark_parameters (dict): Dictionary of benchmark parameters.
        aggregation_functions (dict): Dictionary of aggregation functions for each metric.
        experiment_label (str): Label for the current experiment.

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
    val_df_agg.to_csv(f"{results_dir}/val_results_agg_{experiment_label}.csv")
    test_df_agg.to_csv(f"{results_dir}/test_results_agg_{experiment_label}.csv")
    val_df_agg.to_html(f"{results_dir}/val_results_agg_{experiment_label}.html")
    test_df_agg.to_html(f"{results_dir}/test_results_agg_{experiment_label}.html")

    return val_df_agg, test_df_agg



def save_best_model_weights(source_weights_dir: str, config: dict, model_tag: str, model_config: dict) -> None:
    """
    Save (copy) the best model weights directory into the dedicated 'saved_models' folder.

    Args:
        source_weights_dir (str): Directory of the source model weights.
        config (dict): Experiment configuration dictionary.
        model_tag (str): A string tag to identify the best model (e.g., 'best_test_model_YYYY-MM-DD_HH-MM-SS').
        model_config (dict): Model configuration dictionary.

    Returns:
        None
    """
    #Print the type of the model_config
    logging.info(f"Model configuration type: {type(model_config)}")
    dest_dir = f"experiments/{config['experiment']['project_name']}/saved_models/{model_tag}"
    os.makedirs(dest_dir, exist_ok=True)
    # Use system command to copy the directory (or use shutil.copytree for a cross-platform solution)
    shutil.copytree(source_weights_dir, dest_dir, dirs_exist_ok=True)
    #Save model configuration as well
    try:
        with open(f"{dest_dir}/model_config.json", 'w') as f:
            json.dump(model_config, f)
    except:
        logging.error(f"Failed to save model configuration to {dest_dir}/model_config.json")
        # If the model_config is not serializable, you can use pickle instead
        with open(f"{dest_dir}/model_config.pkl", 'wb') as f:
            pickle.dump(model_config, f)
        logging.warning(f"Saved model configuration as pickle to {dest_dir}/model_config.pkl")
    logging.info(f"Saved best model weights from {source_weights_dir} to {dest_dir}.")
    logging.info(f"Saved best model configuration to {dest_dir}/model_config.json.")


def find_and_apply_best_model(config: dict, val_df_agg: pd.DataFrame, test_df_agg: pd.DataFrame,
                              benchmark_parameters: dict, val_df: pd.DataFrame, test_df: pd.DataFrame,
                              val_dataset: sf.Dataset, test_dataset: sf.Dataset, target: str,
                              slide_level: bool) -> None:
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
        slide_level (bool): Flag indicating if the model is slide-level.

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

    bp_cols = list(benchmark_parameters.keys())
    test_group_cols  = bp_cols + (['test_dataset'] if 'test_dataset' in test_df.columns else [])

    if isinstance(test_df_agg.index, pd.MultiIndex):
        best_test_params_tuple = tuple(best_test_model[:len(bp_cols)])
        best_test_dataset = best_test_model[len(bp_cols)] if 'test_dataset' in test_df.columns else None
    else:
        best_test_params_tuple = best_test_model
        best_test_dataset = None
    
    if isinstance(val_df_agg.index, pd.MultiIndex):
        best_val_params_tuple = tuple(best_val_model[:len(bp_cols)])
    else:
        best_val_params_tuple = best_val_model

    val_mask = val_df[bp_cols].apply(tuple, axis=1) == best_val_params_tuple
    if best_test_dataset is None:
        test_mask_tuple = best_test_params_tuple
    else:
        test_mask_tuple = best_test_params_tuple + (best_test_dataset,)
    test_mask = test_df[test_group_cols].apply(tuple, axis=1) == test_mask_tuple

    # Load best model parameters from the test dataframe
    row = test_df.loc[test_mask].iloc[0]
    with open(row['mil_params'], 'r') as f:
        best_test_model_dict = json.load(f)
        
    best_bag_dir = row['bag_dir']
    if 'task' in best_test_model_dict:
        best_test_model_dict['goal'] = best_test_model_dict['task']
        del best_test_model_dict['task']

    if best_test_model_dict['params']['model'].lower() not in BUILT_IN_MIL:
        if not slide_level and not config['experiment']['aggregation_level'] == 'slide':
            best_test_model_dict['params']['model'] = getattr(aggregators, best_test_model_dict['params']['model'])
        else:
            #If slide-level, use the slide-level predictors
            try:
                mil_method = get_model_class(slide_level_predictors, best_test_model_dict['params']['model'])
            except:
                logging.warning(f"You are either using an undefined slide predictor head or an MIL model: {best_test_model_dict['params']['model']}, Now falling back to Slide-level MLP classifier")
                mil_method = get_model_class(slide_level_predictors, 'mlp_slide_classifier')
                best_test_model_dict['params']['model'] = mil_method
                pass
        best_test_model_config = sf.mil.mil_config(trainer=best_test_model_dict['trainer'], **best_test_model_dict['params'])
    else:
        best_test_model_config = None

    best_test_weights = row['weights']
    logging.info(f"Best test weights directory: {best_test_weights}")

    # Run the best model on the test set
    run_best_model(config, 'test', test_dataset, best_bag_dir,
                   best_test_model_config, target, best_test_weights)

    # Save the best model configuration and copy the weights directory for easy access
    best_model_pickle = f"{saved_models_dir}/best_test_model_{now}.pkl"
    with open(best_model_pickle, 'wb') as f:
        pickle.dump(best_test_model_dict, f)
    save_best_model_weights(best_test_weights, config, f"best_test_model_{now}", best_test_model_dict)


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
        norm="linear"
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
        if 'confusion_matrix' in config['experiment'].get('visualization', []):
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
        metrics['auc'] = metrics['roc_auc_score'] = np.mean(aucs)
        metrics['mean_average_precision'] = np.mean(average_precisions)
        metrics['mean_average_recall'] = np.mean(average_recalls)
        logging.info(f"Classification Metrics - Balanced Accuracy: {metrics['balanced_accuracy']}, Mean F1: {metrics['mean_f1']}, AUC: {metrics['auc']}")

    return metrics, tpr, fpr, precision_recall_data

def _hazards_logits_to_survival_matrix(preds_raw: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Convert discrete-time hazard *logits* with shape (N, K) into survival S(t) on a grid 'times'.
    We assume K equal-width bins spanning [0, t_max], and use sigmoid to map logits→hazards.
    If your preds are already hazards in (0,1), replace the sigmoid with a clip.
    """
    N, K = preds_raw.shape
    hazards = 1.0 / (1.0 + np.exp(-preds_raw))  # sigmoid
    hazards = np.clip(hazards, 1e-9, 1 - 1e-9)
    one_minus_h = 1.0 - hazards
    S_bins = np.cumprod(one_minus_h, axis=1)  # survival at end of each bin, shape (N, K)

    if times.size == 0:
        return np.ones((N, 0), dtype=float)

    t_max = float(np.max(times))
    if t_max <= 0:
        return np.ones((N, len(times)), dtype=float)

    # Map times to bin indices on [0, t_max] with K bins; use step survival of containing bin.
    bin_edges = np.linspace(0.0, t_max, K + 1)
    j_idx = np.searchsorted(bin_edges, times, side="right") - 1
    j_idx = np.clip(j_idx, 0, K - 1)
    return S_bins[:, j_idx]  # (N, T)


def _safe_eval_times(
    durations_test: np.ndarray,
    y_train: tuple | None,
    n_times: int = 100,
) -> np.ndarray:
    """
    Build a time grid strictly inside the *test* follow-up (and intersect with train follow-up
    if provided). This avoids the 'all times must be within follow-up' error.
    """
    d_test = np.asarray(durations_test, float)
    dmin_t, dmax_t = float(np.min(d_test)), float(np.max(d_test))
    if not np.isfinite(dmin_t) or not np.isfinite(dmax_t) or dmax_t <= dmin_t:
        return np.array([], dtype=float)

    eps = (dmax_t - dmin_t) * 1e-6
    lo, hi = dmin_t + eps, dmax_t - eps

    if y_train is not None:
        d_tr, _ = y_train
        d_tr = np.asarray(d_tr, float)
        if d_tr.size:
            tr_min, tr_max = float(np.min(d_tr)), float(np.max(d_tr))
            # intersect (lo, hi) with train range (open interval)
            lo = max(lo, tr_min + eps)
            hi = min(hi, tr_max - eps)

    if hi <= lo:
        return np.array([], dtype=float)

    return np.linspace(lo, hi, n_times)


def _hazards_logits_to_risk(preds_raw: np.ndarray) -> np.ndarray:
    """
    Convert discrete-time hazard *logits* with shape (N, K) into a 1-D risk score.
    We map logits -> hazards via sigmoid, clip for numerical stability,
    then compute cumulative hazard: risk = -sum(log(1 - hazard_k)).
    Larger risk  => higher event risk (worse prognosis).
    """
    logits = preds_raw.astype(np.float64)
    hazards = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    hazards = np.clip(hazards, 1e-9, 1.0 - 1e-9)
    risk = -np.sum(np.log(1.0 - hazards), axis=1)
    return risk


def _surv_labels_from_dataset(ds: sf.Dataset) -> tuple[np.ndarray, np.ndarray]:
    # outcomes=['time','event'] in your pipeline
    lab_dict, _ = ds.labels(['time','event'], format='id')
    arr = np.asarray(list(lab_dict.values()), dtype=float)
    # shape (N, 2): [:,0]=time, [:,1]=event
    durations = arr[:, 0].astype(float)
    events = arr[:, 1].astype(int).astype(bool)
    return durations, events

def _pick_col(result: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Pick first existing column (case-insensitive, substring ok)."""
    cols_lower = {c.lower(): c for c in result.columns}
    for pat in candidates:
        # exact first
        if pat in cols_lower:
            return cols_lower[pat]
    # then substring match
    for c in result.columns:
        lc = c.lower()
        if any(pat in lc for pat in candidates):
            return c
    if required:
        raise ValueError(f"Could not find any of columns: {candidates} in predictions dataframe: {list(result.columns)}")
    return None


def calculate_survival_results(
    result: pd.DataFrame,
    invert_preds: bool = False,
    y_train: tuple | None = None,   # (dur_tr, evt_tr)
    task: str | None = None,        # 'survival' or 'survival_discrete'
    n_times: int = 100,
):
    """
    Robust survival metrics (sksurv-only):
      - Finds durations column among ['time','duration','survival_time'].
      - Finds event column among ['event','y_true','status'].
      - Builds risk so that larger = worse prognosis.
      - Defensive NaN handling with informative errors (no silent all-drop).
      - IPCW c-index made robust (version-agnostic unpack) and restricted
        to a 'safe' censoring window estimated from TRAIN via KM on censoring.
    """
    import numpy as np
    import logging
    from sksurv.util import Surv
    from sksurv.metrics import (
        integrated_brier_score,
        cumulative_dynamic_auc,
        brier_score,
        concordance_index_ipcw,
        concordance_index_censored,
    )
    from sksurv.nonparametric import kaplan_meier_estimator

    # ---- helper: largest train-time with censoring survival G(t) > eps
    def _safe_censor_tail_time_sksurv(dur_tr: np.ndarray, evt_tr: np.ndarray, eps: float = 1e-6) -> float | None:
        try:
            dur = np.asarray(dur_tr, float)
            evt = np.asarray(evt_tr, bool)
            censor_event = ~evt  # KM "event observed" == censoring occurrence
            # KM on censoring; returns times and survival probs Ĝ(t)
            times, G = kaplan_meier_estimator(censor_event.astype(bool), dur.astype(float))
            if times.size == 0 or G.size == 0:
                return None
            ok = G > eps
            if not np.any(ok):
                return None
            return float(times[ok][-1])
        except Exception:
            return None

    # ---- find columns robustly
    dur_col = _pick_col(result, ["time", "duration", "survival_time"])
    evt_col = _pick_col(result, ["event", "y_true", "status"])

    pred_cols = [c for c in result.columns if c.lower().startswith("y_pred")]
    if not pred_cols:
        # fallbacks seen in some pipelines
        pred_cols = [c for c in result.columns if c.lower().startswith("pred")]
    if not pred_cols:
        raise ValueError(f"No prediction columns found (expected y_pred*). Columns: {list(result.columns)}")

    # ---- extract arrays
    durations = result[dur_col].to_numpy(dtype=float)
    # event can be float/str; normalize to {0,1}
    events = (pd.to_numeric(result[evt_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=int)) == 1

    preds_raw = result[pred_cols].to_numpy()

    # ---- risk (larger = worse)
    if preds_raw.ndim == 2 and preds_raw.shape[1] > 1:
        # discrete: per-bin hazard logits -> cumhazard
        risk = _hazards_logits_to_risk(preds_raw.astype(np.float64))
    else:
        # continuous: single score
        risk = np.squeeze(preds_raw.astype(np.float64))
        if invert_preds:
            risk = -risk

    # ---- clean NaNs / infs
    m = np.isfinite(risk) & np.isfinite(durations)
    n_bad = int((~m).sum())
    if n_bad:
        logging.warning(f"Dropping {n_bad} samples with non-finite risk/duration.")
    durations = durations[m]
    events = events[m]
    risk = risk[m]

    if durations.size == 0:
        # diagnostics
        n_dur_nan = int(np.sum(~np.isfinite(result[dur_col].to_numpy(dtype=float))))
        n_pred_nan = int(np.sum(~np.isfinite(np.array(preds_raw, dtype=float))))
        logging.warning(f"All samples dropped! n_dur_nan={n_dur_nan}, n_pred_nan={n_pred_nan}. Cannot compute survival metrics.")
        metrics = {
            "c_index": float("nan"),
            "c_index_ipcw": float("nan"),
            "td_auc_mean": float("nan"),
            "brier_score": float("nan"),
            "ibs": float("nan"),
        }
        return metrics, durations, events.astype(int), np.asarray(risk, dtype=np.float32)
    
    y_test_struct = Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))

    # ---- train struct (optional, for IPCW/TD AUC/IBS)
    if y_train is not None:
        dur_tr, evt_tr = y_train
        dur_tr = np.asarray(dur_tr, float)
        evt_tr = np.asarray(evt_tr, bool)
        mt = np.isfinite(dur_tr)
        if mt.sum() < dur_tr.size:
            logging.warning(f"Dropping {int((~mt).sum())} non-finite TRAIN durations for IPCW/TD-AUC.")
        y_train_struct = Surv.from_arrays(event=evt_tr[mt], time=dur_tr[mt])
    else:
        y_train_struct = None

    # ---- c-index (Harrell)
    c_harrell = float(concordance_index_censored(events, durations, risk)[0])

    # ---- IPCW Uno c-index (sksurv only), trimmed to safe censoring window
    if y_train_struct is not None:
        # compute safe tail time on TRAIN censoring KM
        t_ok = _safe_censor_tail_time_sksurv(dur_tr[mt], evt_tr[mt], eps=1e-6) if y_train is not None else None

        y_test_ipcw = y_test_struct
        risk_ipcw = risk
        if t_ok is not None and np.isfinite(t_ok):
            m_ok = durations <= t_ok
            if not np.all(m_ok):
                logging.debug(
                    f"Restricting IPCW to {int(m_ok.sum())}/{len(m_ok)} samples "
                    f"with t <= {t_ok:.6g} where censor SF > 0."
                )
                y_test_ipcw = Surv.from_arrays(
                    event=events[m_ok].astype(bool),
                    time=durations[m_ok].astype(float)
                )
                risk_ipcw = risk[m_ok]
        try:
            res = concordance_index_ipcw(y_train_struct, y_test_ipcw, risk_ipcw)
            # robust to (est, se), (est, se, ...), or namedtuple
            c_uno = float(res[0]) if isinstance(res, (tuple, list)) else float(res)
        except Exception as ex:
            logging.warning(f"IPCW c-index failed: {ex}; setting NaN.")
            c_uno = float("nan")
    else:
        c_uno = float("nan")

    # ---- time-dependent AUC (sksurv)
    if y_train_struct is not None:
        times = _safe_eval_times(durations, y_train, n_times=n_times)
        if times.size:
            try:
                _, aucs = cumulative_dynamic_auc(y_train_struct, y_test_struct, risk, times)
                td_auc_mean = float(np.nanmean(aucs)) if np.size(aucs) else float("nan")
            except Exception as ex:
                logging.warning(f"cumulative_dynamic_auc failed: {ex}; setting NaN.")
                td_auc_mean = float("nan")
        else:
            td_auc_mean = float("nan")
    else:
        td_auc_mean = float("nan")

    # ---- Brier & IBS (only for discrete-time where we can decode survival curves)
    if preds_raw.ndim == 2 and preds_raw.shape[1] > 1:
        times = _safe_eval_times(durations, y_train, n_times=n_times)
        if times.size:
            try:
                surv_pred = _hazards_logits_to_survival_matrix(preds_raw.astype(np.float64), times)
                _, bs = brier_score(y_train_struct or y_test_struct, y_test_struct, surv_pred, times)
                brier_mean = float(np.nanmean(np.asarray(bs, float)))
            except Exception as ex:
                logging.warning(f"brier_score failed: {ex}; setting NaN.")
                brier_mean = float("nan")
            try:
                ibs = float(integrated_brier_score(y_train_struct or y_test_struct, y_test_struct, surv_pred, times))
            except Exception as ex:
                logging.warning(f"integrated_brier_score failed: {ex}; setting NaN.")
                ibs = float("nan")
        else:
            brier_mean, ibs = float("nan"), float("nan")
    else:
        brier_mean, ibs = float("nan"), float("nan")

    metrics = {
        "c_index": c_harrell,
        "c_index_ipcw": c_uno,
        "td_auc_mean": td_auc_mean,
        "brier_score": brier_mean,
        "ibs": ibs,
    }
    return metrics, durations, events.astype(int), np.asarray(risk, dtype=np.float32)






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

        #Get the dict of additional search parameters in format hp: range e.g. {'epochs' : [10, 50]}
        additional_hyperparameters = config['optimization']['search_parameters'] if 'search_parameters' in config['optimization'] else {}
        if 'epochs' in additional_hyperparameters and len(additional_hyperparameters['epochs']) > 1:
            epochs = trial.suggest_int('epochs', additional_hyperparameters['epochs'][0], additional_hyperparameters['epochs'][1])
        else:
            #If there is one value, use it directly
            epochs = additional_hyperparameters['epochs'][0] if 'epochs' in additional_hyperparameters else config['experiment']['epochs'] if 'epochs' in config['experiment'] else 50

        if 'batch_size' in additional_hyperparameters and len(additional_hyperparameters['batch_size']) > 1:
            batch_size = trial.suggest_int('batch_size', additional_hyperparameters['batch_size'][0], additional_hyperparameters['batch_size'][1])
        else:
            batch_size = additional_hyperparameters['batch_size'][0] if 'batch_size' in additional_hyperparameters else config['experiment']['batch_size'] if 'batch_size' in config['experiment'] else 32

        if 'z_dim' in additional_hyperparameters and len(additional_hyperparameters['z_dim']) > 1:
            z_dim = trial.suggest_int('z_dim', additional_hyperparameters['z_dim'][0], additional_hyperparameters['z_dim'][1])
        else:
            z_dim = additional_hyperparameters['z_dim'][0] if 'z_dim' in additional_hyperparameters else config['experiment']['z_dim'] if 'z_dim' in config['experiment'] else 256
        
        if 'encoder_layers' in additional_hyperparameters and len(additional_hyperparameters['encoder_layers']) > 1:
            encoder_layers = trial.suggest_int('encoder_layers', additional_hyperparameters['encoder_layers'][0], additional_hyperparameters['encoder_layers'][1])
        else:
            encoder_layers = additional_hyperparameters['encoder_layers'][0] if 'encoder_layers' in additional_hyperparameters else config['experiment']['encoder_layers'] if 'encoder_layers' in config['experiment'] else 1
        
        if 'dropout_p' in additional_hyperparameters and len(additional_hyperparameters['dropout_p']) > 1:
            dropout_p = trial.suggest_float('dropout_p', additional_hyperparameters['dropout_p'][0], additional_hyperparameters['dropout_p'][1])
        else:
            dropout_p = additional_hyperparameters['dropout_p'][0] if 'dropout_p' in additional_hyperparameters else config['experiment']['dropout_p'] if 'dropout_p' in config['experiment'] else 0.25

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
        config['experiment']['dropout_p'] = dropout_p

        combination_dict = {
            'tile_px': tile_px,
            'tile_um': tile_um,
            'normalization': normalization,
            'feature_extraction': feature_extraction,
            'mil': mil,
            'loss': loss,
            'augmentation': augmentation,
            'activation_function': activation_function,
            'optimizer': optimizer,
        }

        # Initialize dataframes to record results 
        val_df, test_df = get_column_values(config)

        target = determine_target_variable(task, config)
        annotation_df = pd.read_csv(project.annotations)

        # Prepare the dataset
        all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                   tile_um=combination_dict['tile_um'])
        logging.info("Extracting tiles with quality control...")
        
        qc_list, qc_filters = build_qc_list(config)

        all_data.extract_tiles(enable_downsample=False, save_tiles=config['experiment']['save_tiles'] if 'save_tiles' in config['experiment'] else False,
                               qc=qc_list,
                               grayspace_fraction=float(qc_filters['grayspace_fraction']),
                               whitespace_fraction=float(qc_filters['whitespace_fraction']),
                               grayspace_threshold=float(qc_filters['grayspace_threshold']),
                               whitespace_threshold=int(qc_filters['whitespace_threshold']),
                               num_threads=config['experiment']['num_workers'] if 'num_workers' in config['experiment'] else 1,
                               report=config['experiment']['report'] if 'report' in config['experiment'] else False,
                               skip_extracted=config['experiment']['skip_extracted'] if 'skip_extracted' in config['experiment'] else True,)


        train_datasets, test_datasets = configure_datasets(config)

        train_set, test_set = split_train_test(config, all_data, task)

        splits_file = determine_splits_file(config, project_directory)
        splits = split_datasets(config, project, splits_file, target, project_directory, train_set)
        
        save_string, string_without_mil = get_save_strings(combination_dict)

        logging.debug(f"Save string: {save_string}")

        logging.info("Starting feature extraction...")
        free_up_gpu_memory()
        feature_extractor = build_feature_extractor(name=combination_dict['feature_extraction'].lower(),
                                                    tile_px=combination_dict['tile_px'])
        logging.info("Generating feature bags...")
        bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
        mil_conf, combination_dict = set_mil_config(config, combination_dict, task, slide_level=True if "slide" in combination_dict['feature_extraction'].lower() else False)
        os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
        os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)

        index = 1
        val_results_per_split = []
        val_pr_per_split = []
        survival_results_per_split = []
        test_results_per_split = {ds['name']: [] for ds in test_datasets}
        test_pr_per_split = {ds['name']: [] for ds in test_datasets}
        test_survival_results_per_split = {ds['name']: [] for ds in test_datasets}
        logging.info("Starting training for each split...")
        for train, val in splits:
            logging.info(f"Trial {trial.number} - Split {index} started...")
            train = balance_dataset(train, task, config)
            val = balance_dataset(val, task, config)
            
            if task in ['survival', 'survival_discrete']:
                dur_tr, evt_tr = _surv_labels_from_dataset(train)

            model_kwargs = {
                'pb_config': config,
                'loss': loss,
            }
            # Train the model on the current split
            _ = project.train_mil(
                config=mil_conf,
                outcomes=target,
                train_dataset=train,
                val_dataset=val,
                bags=bags,
                exp_label=f"{save_string}_{index}",
                **model_kwargs
            )

            mil_directory = f"experiments/{config['experiment']['project_name']}/mil"

            number = get_mil_directory_number(mil_directory, save_string)
            def get_validation_metrics(number: str):
                #Get the corresponding validation results
                val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
                # Save the val_result as csv to the mil {number}_{save_string}_{index} directory
                val_result.to_csv(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/val_result.csv", index=False)
                logging.info(f"Validation results saved to experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/val_result.csv")
                if task == 'survival' or task == 'survival_discrete':	
                    metrics, durations, events, predictions = calculate_survival_results(val_result, invert_preds=False, y_train=(dur_tr, evt_tr))
                    survival_results_per_split.append((durations, events, predictions))
                elif task  == 'regression':
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                    y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                    val_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                    val_results_per_split.append([tpr, fpr])
                    val_pr_per_split.extend(val_pr_curves)
                
                return metrics

            metrics = get_validation_metrics(number)

            val_dict = combination_dict.copy()
            val_dict.update(metrics)
            val_df = val_df.append(val_dict, ignore_index=True)


            if test_datasets:
                test_metrics = []
                # If test datasets are provided, evaluate on the test set
                for test_dataset in test_datasets:
                    logging.info(f"Evaluating on test set {test_dataset['name']}...")

                    test_set = all_data.filter(filters={'dataset': [test_dataset['name']]})

                    outdir = (
                    f"{project_directory}/optimization/"
                    f"trial_{trial.number}_"
                    f"{test_dataset['name']}"
                    )

                    test_result = eval_mil(
                        weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}",
                        outcomes=target,
                        dataset=test_set,
                        bags=bags,
                        config=mil_conf,
                        outdir=f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}_{test_dataset['name']}",
                        **model_kwargs
                    )

                    model_string = get_model_string(combination_dict, feature_extraction, config=config)
                    test_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}_{test_dataset['name']}/00000-{model_string}/predictions.parquet")
                    # Save the test_result as csv to the mil_eval {number}_{save_string}_{index} directory
                    test_result.to_csv(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}_{test_dataset['name']}/test_result.csv", index=False)
                    def get_test_metrics():
                        if task in ['survival', 'survival_discrete']:
                            metrics, durations, events, predictions = calculate_survival_results(test_result, invert_preds=False, y_train=(dur_tr, evt_tr))
                            test_survival_results_per_split[test_dataset['name']].append((durations, events, predictions))
                        elif task == 'regression':
                            metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                            y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                            test_results_per_split[test_dataset['name']].append((y_true, y_pred))
                        else:
                            metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                            test_results_per_split[test_dataset['name']].append([tpr, fpr])
                            test_pr_per_split[test_dataset['name']].extend(test_pr_curves)

                        return metrics

                    metrics = get_test_metrics()

                    number = get_mil_directory_number(mil_directory, save_string)

                    test_dict = combination_dict.copy()
                    test_dict.update(metrics)
                    test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
                    test_dict['mil_params'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/mil_params.json"
                    test_dict['bag_dir'] = bags
                    test_dict['test_dataset'] = test_dataset['name']
                    
                    # Append this test result to the overall test results DataFrame
                    test_df = test_df.append(test_dict, ignore_index=True)
            else:
                # If no test datasets are provided, use the validation set as the test set
                logging.warning("No test datasets provided. Using validation set as test set.")
                test_set = val
                # Use the same model weights for evaluation
                test_result = eval_mil(
                    weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}",
                    outcomes=target,
                    dataset=test_set,
                    bags=bags,
                    config=mil_conf,
                    outdir=f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}",
                    **model_kwargs
                )

                model_string = get_model_string(combination_dict, feature_extraction, config=config)
                test_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}/00000-{model_string}/predictions.parquet")
                test_result.to_csv(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}/test_result.csv", index=False)
                def get_test_metrics():
                    if task in ['survival', 'survival_discrete']:
                        metrics, durations, events, predictions = calculate_survival_results(test_result, invert_preds=False)
                        test_survival_results_per_split[test_dataset['name']].append((durations, events, predictions))
                    elif task == 'regression':
                        metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                        y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                        test_results_per_split[test_dataset['name']].append((y_true, y_pred))
                    else:
                        metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                        test_results_per_split[test_dataset['name']].append([tpr, fpr])
                        test_pr_per_split[test_dataset['name']].extend(test_pr_curves)

                    return metrics

                    metrics = get_test_metrics()


                number = get_mil_directory_number(mil_directory, save_string)

                test_dict = combination_dict.copy()
                test_dict.update(metrics)
                test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
                test_dict['mil_params'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/mil_params.json"
                test_dict['bag_dir'] = bags
                test_dict['test_dataset'] = 'validation'
                
                # Append this test result to the overall test results DataFrame
                test_df = test_df.append(test_dict, ignore_index=True)

            logging.info(f"Trial {trial.number} - Split {index} completed.")
            index += 1

        plot_across_splits(config, survival_results_per_split, test_survival_results_per_split,
                           val_results_per_split, test_results_per_split, val_pr_per_split, test_pr_per_split,
                           save_string, invert_preds = True if task == 'survival' else False)

        # Specify which results to use for the objective metric
        if config['optimization']['objective_dataset'] == 'val':
            measure_df = val_df
        elif config['optimization']['objective_dataset'] == 'test':
            measure_df = test_df
        
        if config['optimization']['objective_dataset'] == "test":
            #Aggregate test metrics across datasets
            per_dataset_means = []
            for ds in test_datasets:
                ds_name = ds['name']
                # select only rows for this dataset
                ds_rows = test_df[test_df['test_dataset'] == ds_name]
                if len(ds_rows):
                    per_dataset_means.append(ds_rows[objective_metric].mean())
                else:
                    logging.warning(f"No results for dataset {ds_name}, skipping.")
            if not per_dataset_means:
                raise RuntimeError("No test results to aggregate for Optuna objective!")
            # Final aggregate: mean across dataset‐means
            aggregated_score = float(np.mean(per_dataset_means))

        elif config['optimization']['objective_dataset'] == "val":
            aggregated_score = float(measure_df[objective_metric].mean())

        # Report progress to the trial
        if config['optimization'].get('pruner'):
            trial.report(aggregated_score, index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Save the weights directory for this trial as a user attribute for later retrieval
        trial.set_user_attr("weights_path", test_dict['weights'])
        #Save the model configuration as well
        trial.set_user_attr("mil_params", mil_conf.json_dump())

        logging.info(f"Trial {trial.number} completed with {objective_metric}: {aggregated_score} on {config['optimization']['objective_dataset']} set.")
        return aggregated_score

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
    study.optimize(objective, n_trials=config['optimization']['trials'], show_progress_bar=False)

    # Save the study results
    plot_optimization_output(project_directory, study)
    #Save best parameters
    best_params = study.best_params
    with open(f"{opt_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f)

    logging.info(f"Best parameters: {best_params} found in trial {study.best_trial.number} with value: {study.best_value}")

    #Save best model weights
    # Retrieve and save the best model weights from the best trial
    best_weights = study.best_trial.user_attrs.get("weights_path", None)
    best_conf = study.best_trial.user_attrs.get("mil_params", None)
    #get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if best_weights:
        save_best_model_weights(best_weights, config, f"best_optimized_model_{current_date}", best_conf)
    logging.info("Optimization finished.")


def plot_optimization_output(project_directory, study):
    """
    Plot the optimization output (parameter importances and optimization history).

    Args:
        project_directory (str): Directory for the experiment.
        study (optuna.Study): The Optuna study object.

    Returns:
        None
    """
    opt_dir = f"{project_directory}/optimization"
    os.makedirs(opt_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ax_imp = plot_param_importances(study)
    ax_imp.figure.savefig(f"{opt_dir}/parameter_importances_{ts}.png", bbox_inches="tight")
    ax_imp.figure.clf()

    ax_hist = plot_optimization_history(study)
    ax_hist.figure.savefig(f"{opt_dir}/optimization_history_{ts}.png", bbox_inches="tight")
    ax_hist.figure.clf()

# =============================================================================
# End of Module
# =============================================================================
