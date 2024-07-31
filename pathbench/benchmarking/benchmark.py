from itertools import product
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import softmax
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, brier_score, concordance_index_ipcw
from sksurv.util import Surv
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, average_precision_score, confusion_matrix, roc_curve,
    auc, mean_absolute_error, mean_squared_error, r2_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import sys
import slideflow
import logging
import importlib
import traceback
import pickle
import datetime

from slideflow.model import build_feature_extractor

from slideflow.stats.metrics import ClassifierMetrics

from slideflow.mil import eval_mil, train_mil, mil_config

import slideflow as sf

from ..models.feature_extractors import *
from ..models import aggregators

#Set logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Set default evaluation metrics, based on the task
DEFAULT_METRICS = {
    'classification': ['balanced_accuracy', 'mean_uncertainty', 'auc', 'mean_average_precision', 'mean_average_recall'],
    'regression': ['mean_absolute_error', 'mean_squared_error', 'r2_score'],
    'survival': ['c_index', 'brier_score']
}

#Set essential evaluation metrics, used for sorting the results
ESSENTIAL_METRICS = {
    'classification': ['balanced_accuracy'],
    'regression': ['r2_score'],
    'survival': ['c_index']
}


def get_model_class(module, class_name):
    """
    Get the class from the module based on the class name

    Args:
        module: The module to get the class from
        class_name: The name of the class to get

    Returns:
        The class from the module
    """
    return getattr(module, class_name)


def get_highest_numbered_filename(directory_path : str):
    """
    Get the highest numbered filename in the directory

    Args:
        directory_path: The path to the directory

    Returns:
        The highest numbered filename
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Initialize variables to keep track of the highest number and corresponding filename
    highest_number = float('-inf')
    highest_number_filename = None

    # Iterate over each file
    for filename in files:
        # Get the part before the first '-'
        first_part = filename.split('-')[0]

        # Try to convert the first part to a number
        try:
            number = int(first_part)
            # If the converted number is higher than the current highest, update the variables
            if number > highest_number:
                highest_number = number
                highest_number_part = first_part
        except ValueError:
            pass  # Ignore non-numeric parts

    return highest_number_part



def determine_splits_file(config : dict, project_directory : str):
    """
    Determine the splits file based on the configuration

    Args:
        config: The configuration dictionary
        project_directory: The project directory

    Returns:
        The splits file
    """
    if 'splits' not in config['experiment']:
        #Create default splits file based on experiment name
        splits = ".json"
    else:
        splits = config['experiment']['splits']
    project_directory = f"experiments/{config['experiment']['project_name']}"
    #if os.path.exists(f"{project_directory}/{splits}"):
    task = config['experiment']['task']
    if config['experiment']['split_technique'] == 'fixed':
        splits_file = f"{project_directory}/fixed_{task}_{splits}"
    elif config['experiment']['split_technique'] == 'k-fold':
        splits_file = f"{project_directory}/kfold_{task}_{splits}"
    else:
        logging.error("Invalid split technique. Please choose either 'fixed' or 'k-fold'")
        sys.exit(1)
    return splits_file


def set_metrics(config : dict):
    """
    Set the evaluation metrics based on the configuration

    Args:
        config: The configuration dictionary
    
    Returns:
        The updated configuration, evaluation metrics and aggregation functions
    """
    # Use default metrics if none are specified in the config
    if 'evaluation' not in config['experiment'] or not config['experiment']['evaluation']:
        config['experiment']['evaluation'] = DEFAULT_METRICS[config['experiment']['task']]
    else:
        # Ensure essential metrics are always included
        essential_metrics = ESSENTIAL_METRICS[config['experiment']['task']]
        config['experiment']['evaluation'] = list(set(config['experiment']['evaluation'] + essential_metrics))
    #Set the evaluation metrics
    evaluation_metrics = config['experiment']['evaluation']

    #Set the aggregation functions for these metrics
    aggregation_functions = {metric: ['mean', 'std'] for metric in evaluation_metrics}

    # Filter out non-relevant metrics based on the task
    relevant_metrics = DEFAULT_METRICS[config['experiment']['task']]

    # Only keep the relevant metrics
    aggregation_functions = {metric: agg for metric, agg in aggregation_functions.items() if metric in relevant_metrics}

    return config, evaluation_metrics, aggregation_functions


def calculate_combinations(config : dict):
    """
    Calculate all combinations based on the benchmark parameters in the configuration

    Args:
        config: The configuration dictionary

    Returns:
        The list of all combinations
    """
    # Retrieve all combinations
    benchmark_parameters = config['benchmark_parameters']
    combinations = []
    for values in benchmark_parameters.values():
        if isinstance(values, list):
            combinations.append(values)

    all_combinations = list(product(*combinations))
    return all_combinations


def should_continue(config : dict, all_combinations : list):
    """
    Check if the benchmarking should continue based on the finished combinations

    Args:
        config: The configuration dictionary
        all_combinations: The list of all combinations

    Returns:
        A boolean indicating whether the benchmarking should continue, the filtered list of all combinations, the validation results and the test results
        all_combinations: The filtered list of all combinations
        val_df: The validation results dataframe
        test_df: The test results dataframe
    """
    finished_file_path = f"{config['experiment']['project_name']}/finished_combinations.pkl"
    
    # Ensure the finished_combinations.pkl file exists and is not empty
    if not os.path.exists(finished_file_path):
        with open(finished_file_path, 'wb') as f:
            pickle.dump([], f)  # Use a list instead of a dictionary for combinations
    
    # Load finished combinations
    if os.path.exists(finished_file_path) and os.path.getsize(finished_file_path) > 0:
        with open(finished_file_path, 'rb') as f:
            finished_combinations = pickle.load(f)
    else:
        finished_combinations = []

    # Filter all_combinations
    all_combinations = [x for x in all_combinations if x not in finished_combinations]

    #If all combinations are finished, redo the benchmarking
    if len(all_combinations) == 0:
        finished_combinations = []
        with open(finished_file_path, 'wb') as f:
            pickle.dump(finished_combinations, f)
        all_combinations = list(product(*combinations))
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
    else:
        # Load validation and test results if they exist
        val_results_path = f"{config['experiment']['project_name']}/results/val_results.csv"
        test_results_path = f"{config['experiment']['project_name']}/results/test_results.csv"
        
        if os.path.exists(val_results_path):
            val_df = pd.read_csv(val_results_path)
        else:
            val_df = pd.DataFrame()
        
        if os.path.exists(test_results_path):
            test_df = pd.read_csv(test_results_path)
        else:
            test_df = pd.DataFrame()

    return True if len(all_combinations) > 0 else False, all_combinations, val_df, test_df


def determine_target_variable(task : str, config : dict):
    """
    Determine the target variable based on the task

    Args:
        task: The task
        config: The configuration dictionary

    Returns:
        The target variable
    """
    if task == 'survival':
        target = ['time', 'event']
    elif task == 'classification':
        target = 'category'
    elif task == 'regression':
        target = 'value'
    else:
        logging.error("Invalid task. Please choose either 'survival', 'classification' or 'regression'")
        sys.exit(1)
    return target


def balance_dataset(dataset : sf.Dataset, task: str, config : dict):
    """
    Determine the balancing strategy based on the task amd apply balancing

    Args:
        dataset: The training dataset
        task: The task
        config: The configuration dictionary

    Returns:
        The balanced dataset
    """
    if task == 'survival':
        headers = ['event']
        logging.info(f"Balancing datasets on {headers} for training using {config['experiment']['balancing']} strategy")
        dataset.balance(headers=headers, strategy=config['experiment']['balancing'], force=True)
    elif task == 'classification':
        headers = ['category']
        logging.info(f"Balancing datasets on {headers} for training using {config['experiment']['balancing']} strategy")
        dataset.balance(headers=headers, strategy=config['experiment']['balancing'], force=True)
    else:
        logging.info("No balancing needed for regression task")

    return dataset



def split_datasets(config : dict, project : sf.Project, splits_file : str, target : str, project_directory : str,
                   train_set : sf.Dataset):
    """
    Split the datasets based on the configuration

    Args:
        config: The configuration dictionary
        project: The project
        splits_file: The splits file
        target: The target variable
        project_directory: The project directory
        train_set: The training dataset

    Returns:
        The splits
    """
    if config['experiment']['task'] == 'classification':
        target = "category"
        model_type = 'categorical'
    elif config['experiment']['task'] == 'survival':
        target = "event"
        model_type = 'categorical'
    elif config['experiment']['task'] == 'regression':
        target = None
        model_type = 'linear'
    if config['experiment']['split_technique'] == 'k-fold':
        k = config['experiment']['k']
        if os.path.exists(f"{project_directory}/kfold_{splits_file}"):
            splits = train_set.kfold_split(k=k, labels=target,
                                        splits=splits_file, read_only=True)
        else:
            splits = train_set.kfold_split(k=k, labels=target, splits=splits_file)
            logging.info(f"K-fold splits have been rewritten to {project_directory}/kfold_{splits}")
    else:
        if os.path.exists(f"{project_directory}/fixed_{splits}"):
            train, val = train_set.split(labels=target,
                                    model_type = model_type,
                                    val_strategy=config['experiment']['split_technique'],
                                    val_fraction=config['experiment']['val_fraction'],
                                    splits=splits_file,
                                    read_only=True)
        else:
            train, val = train_set.split(labels=target,
                                    model_type = model_type,
                                    val_strategy=config['experiment']['split_technique'],
                                    val_fraction=config['experiment']['val_fraction'],
                                    splits=splits_file)
            logging.info(f"Fixed splits have been rewritten to {project_directory}/fixed_{splits}")
        splits = [(train, val)]
    return splits


def generate_bags(config : dict, project : sf.Project, all_data : sf.Dataset,
                  combination_dict : dict, string_without_mil : str,
                  feature_extractor):
    """
    Generate feature bags based on the configuration

    Args:
        config: The configuration dictionary
        project: The project
        all_data: The dataset
        combination_dict: The combination dictionary
        string_without_mil: The string without the MIL method
        feature_extractor: The feature extractor

    Returns:
        The feature bags
    """
    #Generate feature bags
    os.makedirs(f"{config['experiment']['project_name']}/bags", exist_ok=True)
    if os.path.exists(f"config['experiment']['project_name']/bags/{string_without_mil}"):
        bags = f"{config['experiment']['project_name']}/bags/{string_without_mil}"
    else:
        bags = project.generate_feature_bags(model=feature_extractor, 
                                            dataset= all_data,
                                            normalizer=combination_dict['normalization'],
                                            outdir=f"experiments/{config['experiment']['project_name']}/bags/{string_without_mil}")
    return bags


def set_mil_config(config : dict, combination_dict : dict):
    """
    Set the MIL configuration based on the configuration

    Args:
        config: The configuration dictionary
        combination_dict: The combination dictionary

    Returns:
        The MIL configuration
    """
    built_in_mil = ['clam_sb', 'clam_mb', 'attention_mil', 'mil_fc', 'mil_fc_mc', 'transmil', 'bistro.transformer']
    #Check for non-slideflow MIL method
    if combination_dict['mil'].lower() not in built_in_mil:
        mil_method = get_model_class(aggregators, combination_dict['mil'].lower())
        mil_conf = mil_config(mil_method, aggregation_level=config['experiment']['aggregation_level'], trainer="fastai",
                            epochs=config['experiment']['epochs'], drop_last=False)
    else:
        mil_method = combination_dict['mil'].lower()
        #Check whether multiclass and CLAM problem
        mil_conf = mil_config(mil_method, aggregation_level=config['experiment']['aggregation_level'], trainer="fastai",
                    epochs=config['experiment']['epochs'], drop_last=False)
    return mil_conf


def plot_across_splits(config : dict, survival_results_per_split : list, test_survival_results_per_split : list,
                        val_results_per_split : list, test_results_per_split : list, val_pr_per_split : list, test_pr_per_split : list,
                        save_string : str):
    """
    Plot the results across splits, depending on the task

    Args:
        config: The configuration dictionary
        survival_results_per_split: The survival results per split, only relevant for survival task
        test_survival_results_per_split: The test survival results per split, only relevant for survival task
        val_results_per_split: The validation results per split
        test_results_per_split: The test results per split
        val_pr_per_split: The validation precision-recall curves per split, only relevant for classification
        test_pr_per_split: The test precision-recall curves per split, only relevant for classification
        save_string: The save string

    Returns:
        None
    """
    if config['experiment']['task'] == 'survival':
        plot_survival_auc_across_folds(survival_results_per_split, save_string, 'val', config)
        plot_survival_auc_across_folds(test_survival_results_per_split, save_string, 'test', config)
        plot_concordance_index_across_folds(survival_results_per_split, save_string, 'val', config)
        plot_concordance_index_across_folds(test_survival_results_per_split, save_string, 'test', config)
        plot_calibration_across_splits(survival_results_per_split, save_string, 'val', config)
        plot_calibration_across_splits(test_survival_results_per_split, save_string, 'test', config)
    
    elif config['experiment']['task'] == 'regression':
        plot_residuals_across_folds(val_results_per_split, save_string, 'val', config)
        plot_residuals_across_folds(test_results_per_split, save_string, 'test', config)
        plot_predicted_vs_actual_across_folds(val_results_per_split, save_string, 'val', config)
        plot_predicted_vs_actual_across_folds(test_results_per_split, save_string, 'test', config)

    elif config['experiment']['task'] == 'classification':    
        plot_roc_curve_across_splits(val_results_per_split, save_string, "val", config)
        plot_roc_curve_across_splits(test_results_per_split, save_string, "test", config)
        plot_precision_recall_curve_across_splits(val_pr_per_split, save_string, "val", config)
        plot_precision_recall_curve_across_splits(test_pr_per_split, save_string, "test", config)


def build_aggregated_results(val_df : pd.DataFrame, test_df: pd.DataFrame,
                             benchmark_parameters : dict, aggregation_functions : dict):
    """
    Aggregate the results across splits, based on the relevant metric for each task

    Args:
        val_df: The validation results dataframe
        test_df: The test results dataframe
        benchmark_parameters: The benchmark parameters
        aggregation_functions: The aggregation functions

    Returns:
        The aggregated validation results dataframe and the aggregated test results dataframe
    """
    
    # Group dataframe and aggregate
    val_df_grouped = val_df.groupby(list(benchmark_parameters.keys()))
    test_df_grouped = test_df.groupby(list(benchmark_parameters.keys()))

    val_df_agg = val_df_grouped.agg(aggregation_functions)
    test_df_agg = test_df_grouped.agg(aggregation_functions)

    val_df_agg.columns = ['_'.join(col).strip() for col in val_df_agg.columns.values]
    test_df_agg.columns = ['_'.join(col).strip() for col in test_df_agg.columns.values]

    # Sort the dataframes based on the relevant metric for each task
    if config['experiment']['task'] == 'classification':
        val_df_agg = val_df_agg.sort_values(by='balanced_accuracy_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='balanced_accuracy_mean', ascending=False)
    elif config['experiment']['task'] == 'regression':
        val_df_agg = val_df_agg.sort_values(by='r2_score_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='r2_score_mean', ascending=False)
    elif config['experiment']['task'] == 'survival':
        val_df_agg = val_df_agg.sort_values(by='c_index_mean', ascending=False)
        test_df_agg = test_df_agg.sort_values(by='c_index_mean', ascending=False)

    # Save all dataframes
    os.makedirs(f"{config['experiment']['project_name']}/results", exist_ok=True)
    val_df_agg.to_csv(f"{config['experiment']['project_name']}/results/val_results_agg.csv")
    test_df_agg.to_csv(f"{config['experiment']['project_name']}/results/test_results_agg.csv")

    # Save the dataframes as HTML
    val_df_agg.to_html(f"{config['experiment']['project_name']}/results/val_results_agg.html")
    test_df_agg.to_html(f"{config['experiment']['project_name']}/results/test_results_agg.html")

    return val_df_agg, test_df_agg


def find_and_apply_best_model(config : dict, val_df_agg : pd.DataFrame, test_df_agg : pd.DataFrame,
                              benchmark_parameters : dict, val_df : pd.DataFrame, test_df : pd.DataFrame):
    """
    Find and apply the best model based on the relevant metric for each task

    Args:
        config: The configuration dictionary
        val_df_agg: The aggregated validation results dataframe
        test_df_agg: The aggregated test results dataframe
        benchmark_parameters: The benchmark parameters
        val_df: The validation results dataframe
        test_df: The test results dataframe

    Returns:
        None
    """
    # Select best performing model from both validation and test, based on the relevant metric
    if config['experiment']['task'] == 'classification':
        best_val_model = val_df_agg['balanced_accuracy_mean'].idxmax()
        best_test_model = test_df_agg['balanced_accuracy_mean'].idxmax()
    elif config['experiment']['task'] == 'regression':
        best_val_model = val_df_agg['r2_score_mean'].idxmax()
        best_test_model = test_df_agg['r2_score_mean'].idxmax()
    elif config['experiment']['task'] == 'survival':
        best_val_model = val_df_agg['c_index_mean'].idxmax()
        best_test_model = test_df_agg['c_index_mean'].idxmax()

    logging.info(f"Best validation model: {best_val_model}")
    logging.info(f"Best test model: {best_test_model}")

    # Save best models
    os.makedirs(f"experiments/{config['experiment']['project_name']}/saved_models", exist_ok=True)

    # Get current date
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Save the best models data
    val_df.loc[val_df[benchmark_parameters.keys()].apply(tuple, 1) == best_val_model].to_csv(f"experiments/{config['experiment']['project_name']}/saved_models/best_val_model_{date_string}.csv")
    test_df.loc[test_df[benchmark_parameters.keys()].apply(tuple, 1) == best_test_model].to_csv(f"experiments/{config['experiment']['project_name']}/saved_models/best_test_model_{date_string}.csv")

    # Load best model df without header
    best_val_model_df = pd.read_csv(f"experiments/{config['experiment']['project_name']}/saved_models/best_val_model_{date_string}.csv", header=None)
    best_test_model_df = pd.read_csv(f"experiments/{config['experiment']['project_name']}/saved_models/best_test_model_{date_string}.csv", header=None)

    #Get value from series
    best_weights = test_df['weights']
    best_weights = best_weights[best_weights.index[0]]

    run_best_model(config, 'val', val, bags, mil_conf, target, best_weights)
    run_best_model(config, 'test', test, bags, mil_conf, target, best_weights)

    # Load the configuration as a dictionary
    best_val_model_dict = dict(zip(list(benchmark_parameters.keys()), best_val_model_df.values[0]))
    best_test_model_dict = dict(zip(list(benchmark_parameters.keys()), best_test_model_df.values[0]))

    print(best_val_model_dict)
    print(best_test_model_dict)

    # Save the best configurations
    with open(f"experiments/{config['experiment']['project_name']}/saved_models/best_val_model_{date_string}.pkl", 'wb') as f:
        pickle.dump(best_val_model_dict, f)
    with open(f"experiments/{config['experiment']['project_name']}/saved_models/best_test_model_{date_string}.pkl", 'wb') as f:
        pickle.dump(best_test_model_dict, f)

    string_to_search = f"{best_test_model_dict['tile_px']}_{best_test_model_dict['tile_um']}_{best_test_model_dict['normalization']}_{best_test_model_dict['feature_extraction']}_{best_test_model_dict['mil']}"
    dir_to_search = f"experiments/{config['experiment']['project_name']}/mil"
    # Get the subdirectories which contain the string
    subdirs = [f.path for f in os.scandir(dir_to_search) if f.is_dir() and string_to_search in f.path]
    # Copy these directories into a best model directory
    os.makedirs(f"experiments/{config['experiment']['project_name']}/saved_models/best_model_{date_string}", exist_ok=True)
    for subdir in subdirs:
        shutil.copytree(subdir, f"experiments/{config['experiment']['project_name']}/saved_models/best_model_{date_string}/{os.path.basename(subdir)}")


def benchmark(config, project):
    """
    Main benchmarking script, which runs the benchmarking based on the configuration

    Args:
        config: The configuration dictionary
        project: The project

    Returns:
        None
    """
    #Set the splits
    logging.info("Starting benchmarking...")
    task = config['experiment']['task']
    benchmark_parameters = config['benchmark_parameters']
    
    project_directory = f"experiments/{config['experiment']['project_name']}"
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

    if config['experiment']['with_continue']:
        to_continue, all_combinations, val_df, test_df = should_continue(config, all_combinations)
        if not to_continue:
            logging.info("All combinations have been finished. Restarting benchmarking...")
        else:
            logging.info(f"Continuing, {len(all_combinations)} combinations left...")

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

            annotation_df = pd.read_csv(project.annotations)
            n_class = len(annotation_df[target].unique())

            #Split datasets into train, val and test
            all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                    tile_um=combination_dict['tile_um'],
                                    )
            
            logging.info("Extracting tiles...")
            #Extract tiles with QC for all datasets
            all_data.extract_tiles(enable_downsample=False,
                                    save_tiles=False,
                                    qc="both",
                                    skip_extracted=True)
                                
            train_set = all_data.filter(filters={'dataset' : 'train'})

            #Balance the training dataset
            train_set = balance_dataset(train_set, task, config)

            test_set = all_data.filter(filters={'dataset' : 'validate'})

            logging.info("Splitting datasets...")
            splits = split_datasets(config, project, splits_file, target, project_directory, train_set)

            save_string = "_".join([f"{value}" for value in combination_dict.values()])
            string_without_mil = "_".join([f"{value}" for key, value in combination_dict.items() if key != 'mil'])
            
            logging.debug(f"Save string: {save_string}") 
            #Run with current parameters
            
            logging.info("Feature extraction...")
            feature_extractor = build_feature_extractor(combination_dict['feature_extraction'].lower(),
                                                        tile_px=combination_dict['tile_px'])
            


            logging.info("Training MIL model...")
            #Generate bags
            bags = generate_bags(config, project, all_data, combination_dict, string_without_mil, feature_extractor)
            #Set MIL configuration
            mil_conf = set_mil_config(config, combination_dict)

            index = 1
            val_results_per_split, test_results_per_split = [], []
            val_pr_per_split, test_pr_per_split = [], []
            survival_results_per_split, test_survival_results_per_split = [], []
            logging.info("Starting training...")
            for train, val in splits:
                logging.info(f"Split {index} started...")

                val_result = project.train_mil(
                    config=mil_conf,
                    outcomes=target,
                    train_dataset=train,
                    val_dataset=val,
                    bags=bags,
                    exp_label=f"{save_string}_{index}",
                    task=config['experiment']['task'],
                )
                #Get current newest MIL model number
                number = get_highest_numbered_filename(f"experiments/{config['experiment']['project_name']}/mil/")
                #Get the corresponding validation results
                val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}/predictions.parquet")
                #Print the unique values in y_pred0:
                print(val_result)
                if config['experiment']['task'] == 'survival':
                    metrics, durations, events, predictions = calculate_survival_results(val_result, config, save_string)
                    survival_results_per_split.append((durations, events, predictions))
                elif config['experiment']['task'] == 'regression':
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string)
                    y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                    val_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string)
                    val_results_per_split.append([tpr, fpr])
                    val_pr_per_split.append(val_pr_curves)
                
                if config['experiment']['task'] == 'classification':
                    calculate_uncertainty(val_result, save_string, "val", config)
                val_dict = combination_dict.copy()
                val_dict.update(metrics)

                val_df = val_df.append(val_dict, ignore_index=True)
                # Test the trained model
                test_result = eval_mil(
                    weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}",
                    outcomes=target,
                    dataset=test_set,
                    bags=bags,
                    config=mil_conf,
                    outdir=f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}",
                    task=config['experiment']['task'],
                )   
                if combination_dict['mil'].lower() in built_in_mil:
                    model_string = combination_dict['mil'].lower()
                else:
                    model_string = f"<class 'pathbench.models.aggregators.{combination_dict['mil'].lower()}'>"
                test_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-{save_string}_{index}/00000-{model_string}/predictions.parquet")
                print(test_result)
                if config['experiment']['task'] == 'survival':
                    metrics, durations, events, predictions = calculate_survival_results(test_result, config, save_string)
                    test_survival_results_per_split.append((durations, events, predictions))
                elif config['experiment']['task'] == 'regression':
                    metrics, tpr, fprr, test_pr_curves = calculate_results(test_result, config, save_string)
                    y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                    test_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string)
                    test_results_per_split.append([tpr, fpr])
                    test_pr_per_split.append(test_pr_curves)
                if config['experiment']['task'] == 'classification':
                    calculate_uncertainty(test_result, save_string, "test", config)
                test_dict = combination_dict.copy()
                test_dict.update(metrics)
                #Add weights directory to test dict
                test_dict['weights'] = f"experiments/{config['experiment']['project_name']}/mil/{number}-{save_string}_{index}"
                
                if 'umap' in config['experiment']['visualization'] or 'mosaic' in config['experiment']['visualization']:
                    visualize_activations(config, val, f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}", bags, target, save_string)
                    visualize_activations(config, test, f"experiments/{config['experiment']['project_name']}/tfrecords/{combination_dict['tile_px']}px_{combination_dict['tile_um']}", bags, target, save_string)

                test_df = test_df.append(test_dict, ignore_index=True)
                print(test_df)
                index += 1

            logging.info(f"Combination {save_string} finished...")
            # Save which combinations were finished
            finished_combination_dict[combination] = 1
            with open(f"experiments/{config['experiment']['project_name']}/finished_combinations.pkl", 'wb') as f:
                pickle.dump(finished_combination_dict, f)

            # Save the combination results up to this point, and mark it as finished
            val_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/val_results.csv")
            test_df.to_csv(f"experiments/{config['experiment']['project_name']}/results/test_results.csv")

            # Check if there was more than one split
            if len(splits) > 1:
                plot_across_splits(config, survival_results_per_split, test_survival_results_per_split,
                                    val_results_per_split, test_results_per_split, val_pr_per_split, test_pr_per_split,
                                    save_string)
        except Exception as e:
            logging.warning(f"Combination {save_string} was not succesfully trained due to Error {e}")
            logging.warning(traceback.format_exc())

    print(val_df, test_df)
    print(list(benchmark_parameters.keys()))

    val_df_agg, test_df_agg = build_aggregated_results(val_df, test_df, benchmark_parameters, aggregation_functions)
    
    find_and_apply_best_model(config, val_df_agg, test_df_agg, benchmark_parameters, val_df, test_df)
    
    # Remove the finished combinations
    with open(f"experiments/{config['experiment']['project_name']}/finished_combinations.pkl", 'wb') as f:
        pickle.dump({}, f)

    # Empty the val and test results
    os.remove(f"experiments/{config['experiment']['project_name']}/results/val_results.csv")
    os.remove(f"experiments/{config['experiment']['project_name']}/results/test_results.csv")
    logging.info("Benchmarking finished...")



def run_best_model(config: dict, split : int, dataset : sf.Dataset, bags : str, mil_conf : dict, target : str, model_weights : str):
    """
    Run the best model based on the configuration, including heatmaps

    Args:
        config: The configuration dictionary
        split: The split
        dataset: The dataset
        bags: The bags
        mil_conf: The MIL configuration
        target: The target variable
        model_weights: Path to the model weights
    
    Returns:
        None
    """
    #Run the best model and generate heatmap
    logging.info("Running the best model...")
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result = eval_mil(
        weights=model_weights,
        outcomes=target,
        dataset=dataset,
        bags=bags,
        config=mil_conf,
        outdir=f"experiments/{config['experiment']['project_name']}/best_model_eval_{split}_{date}",
        task=config['experiment']['task'],
        attention_heatmaps=True,
        cmap="jet",
        norm="linear"
    )


def visualize_activations(config : dict, dataset : sf.Dataset,
                          tfrecord_dir : str, bag_dir : str, target : str, save_string : str):
    """
    Visualize the activations based on the configuration

    Args:
        config: The configuration dictionary
        dataset: The dataset
        tfrecord_dir: Directory with tfrecords
        bag_dir: Directory with bags
        target: The target variable
        save_string: The save string
    
    Returns:
        None

    """

    
    dts_ftrs = sf.DatasetFeatures.from_bags(bag_dir)

    slide_map = sf.SlideMap.from_features(dts_ftrs)

    if 'umap' in config['experiment']['visualization'] or 'mosaic' in config['experiment']['visualization']:
        logging.info("Visualizing activations...")
        labels, unique_labels = dataset.labels(target, format='name')
        for index, label in enumerate(unique_labels):
            try:
                #TOFIX: DOES NOT WORK NOW
                slide_map.label_by_preds(index=index)
                slide_map.save_plot(
                    filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_pred_{save_string}_{index}",
                    title=f"Feature UMAP, by prediction of class {index}"
                )

                plt.close() 
            except:
                print(f"Could not visualize UMAP for class {index}")
                pass
            
        slide_map.label_by_slide(labels)
        #Make a new directory inside visualizations
        if not os.path.exists(f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}"):
            os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}")
        slide_map.save_plot(
            filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}",
            title="Feature UMAP, by label",
            subsample=2000
        )

        plt.close()
    
    if 'mosaic' in config['experiment']['visualization']:

        #TOFIX: DOES NOT WORK NOW!
        logging.info("Building mosaic...")
        # Get list of all directories in the tfrecords dir with full path
        dir_list = [os.path.join(tfrecord_dir, x) for x in os.listdir(tfrecord_dir) if x.endswith('.tfrecord')]
        print(dir_list)
        mosaic = slide_map.build_mosaic(tfrecords=dir_list)
        mosaic.save(
            filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_mosaic_{save_string}.png"
        )
        plt.close()


"""
def augment_bags(bags, config):
    aug_prob = config['experiment']['aug_prob']
    augmentations = config['benchmark_parameters']['aug']
    logging.info(f"Augmenting bags with {augmentations}, probability: {aug_prob}...")
    
    # Load the .pt bags
    for bag in bags:
        # Load the original features from the bag
        original_features = torch.load(bag)
        
        # Save the original features with '_original' suffix
        original_bag_name = os.path.splitext(bag)[0] + '_original.pt'
        torch.save(original_features, original_bag_name)
        
        # Initialize augmented_features as the original features
        augmented_features = original_features.clone()
        augmented = False  # Flag to track if any augmentation is applied

        # Apply augmentations
        for aug in augmentations:
            # With a probability of aug_prob, apply the augmentations
            if np.random.rand() > aug_prob:
                continue
            else:
                if aug == 'patch_dropout':
                    augmented_features = patch_dropout(augmented_features)
                elif aug == 'shuffle_patches':
                    augmented_features = shuffle_patches(augmented_features)
                elif aug == 'add_gaussian_noise':
                    augmented_features = add_gaussian_noise(augmented_features)
                elif aug == 'random_scaling':
                    augmented_features = random_scaling(augmented_features)
                elif aug == 'feature_masking':
                    augmented_features = feature_masking(augmented_features)
                elif aug == 'feature_dropout':
                    augmented_features = feature_dropout(augmented_features)
                augmented = True

        # Save the augmented bag with the original filename
        # If no augmentation took place, save the original features
        if augmented:
            torch.save(augmented_features, bag)
            logging.info(f"Augmented bag saved to {bag}")
        else:
            # If no augmentation took place, save the original features to the original name
            torch.save(original_features, bag)
            logging.info(f"No augmentation applied to {bag}")
"""

def calculate_results(result : pd.DataFrame, config : dict, save_string : str):
    """
    Calculate the results based on the configuration, given the results dataframe and the
    task.

    Args:
        result: The results dataframe
        config: The configuration dictionary
        save_string: The save string

    Returns:
        The metrics, true positive rate, false positive rate and precision-recall curves
    """
    metrics = {}
    y_pred_cols = [c for c in result.columns if c.startswith('y_pred')]

    task = config['experiment']['task']
    # Calculate uncertainty for each row if task is classification
    if task == 'classification':
        result['uncertainty'] = result.apply(calculate_entropy, axis=1)
        mean_uncertainty = result['uncertainty'].mean()
        metrics['mean_uncertainty'] = mean_uncertainty

    # Initialize metric lists
    balanced_accuracies = []
    aucs = []
    average_precisions = []
    average_recalls = []
    f1_scores = {}

    precision_recall_data = []

    tpr, fpr = [], []
    
    if task == 'regression':
        # Calculate regression metrics
        y_true = result['y_true'].values
        y_pred = result['y_pred0'].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics['mean_absolute_error'] = mae
        metrics['mean_squared_error'] = mse
        metrics['r2_score'] = r2

        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"R² Score: {r2}")

    else:  # classification and multiclass classification
        all_y_true = []
        all_y_pred = []
        unique_classes = np.unique(result.y_true.values)

        for idx in range(len(y_pred_cols)):
            y_true_binary = (result.y_true.values == unique_classes[idx]).astype(int)
            y_pred = result[f'y_pred{idx}'].values

            m = ClassifierMetrics(y_true=y_true_binary, y_pred=y_pred)
            fpr, tpr, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = threshold[optimal_idx]
            y_pred_binary_opt = (y_pred > optimal_threshold).astype(int)

            balanced_accuracy = balanced_accuracy_score(y_true_binary, y_pred_binary_opt)
            f1 = f1_score(y_true_binary, y_pred_binary_opt)

            balanced_accuracies.append(balanced_accuracy)
            aucs.append(auroc)
            f1_scores[f'f1_score_{idx}'] = f1

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
            average_precision = average_precision_score(y_true_binary, y_pred)

            average_precisions.append(average_precision)
            average_recalls.append(recall.mean())

            precision_recall_data.append((precision, recall))

            # Collect all true and predicted values
            all_y_true.extend(result.y_true.values[result.y_true.values == unique_classes[idx]])
            all_y_pred.extend([unique_classes[idx]] * sum(y_pred_binary_opt))

            # Save correct and incorrect classifications per slide
            os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
            correct = result[(result.y_true.values == unique_classes[idx])]
            incorrect = result[(result.y_true.values != unique_classes[idx])]
            correct.to_csv(f"experiments/{config['experiment']['project_name']}/results/correct_{save_string}_{idx}.csv")
            incorrect.to_csv(f"experiments/{config['experiment']['project_name']}/results/incorrect_{save_string}_{idx}.csv")

            # Compute and plot the confusion matrix for individual classes
            cm = confusion_matrix(y_true_binary, y_pred_binary_opt)
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for Class {unique_classes[idx]}")
            os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
            plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/confusion_matrix_{save_string}_{idx}.png")
            plt.close()

            # Plot ROC-AUC curve
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color="orange", lw=lw, label=f"ROC curve (area = %0.2f)" % auroc)
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC-AUC")
            plt.legend(loc="lower right")
            plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/roc_auc_{save_string}_{idx}.png")
            plt.close()

            # Plot precision-recall curve
            plt.figure()
            plt.plot(recall, precision, color="b", lw=lw, label=f"Precision-Recall curve (area = %0.2f)" % average_precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/precision_recall_{save_string}_{idx}.png")
            plt.close()

            logging.info(f"Results for combination {save_string}:")
            logging.info(f"Balanced accuracy: {balanced_accuracy}")
            logging.info(f"AUC: {auroc}")
            logging.info(f"Average Precision: {average_precision}")
            logging.info(f"F1 Score: {f1}")

        # Overall confusion matrix
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        cm_overall = confusion_matrix(all_y_true, all_y_pred, labels=unique_classes)
        cm_overall_percentage = cm_overall.astype('float') / cm_overall.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_overall_percentage, display_labels=unique_classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Overall Confusion Matrix")
        plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/confusion_matrix_overall_{save_string}.png")
        plt.close()

        metrics['balanced_accuracy'] = np.mean(balanced_accuracies)
        metrics.update(f1_scores)
        metrics['auc'] = np.mean(aucs)
        metrics['mean_average_precision'] = np.mean(average_precisions)
        metrics['mean_average_recall'] = np.mean(average_recalls)

        logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
        for idx in range(len(y_pred_cols)):
            logging.info(f"F1 Score for class {unique_classes[idx]}: {metrics[f'f1_score_{idx}']}")
        logging.info(f"Mean AUC: {metrics['auc']}")
        logging.info(f"Mean Average Precision: {metrics['mean_average_precision']}")
        logging.info(f"Mean Average Recall: {metrics['mean_average_recall']}")

    return metrics, tpr, fpr, precision_recall_data


def calculate_survival_results(result : pd.DataFrame, config : dict, save_string : str):
    """
    Calculate the survival results based on the configuration, given the results dataframe

    Args:
        result: The results dataframe
        config: The configuration dictionary
        save_string: The save string

    Returns:
        The metrics, durations, events and predictions
    """
    logging.info("Calculating survival metrics...")
    print(result)
    metrics = {}
    durations = result['duration'].values
    events = result['y_true'].values
    predictions = result['y_pred0'].values  # Assuming single column prediction for simplicity

    # Calculate C-index
    c_index = concordance_index(durations, predictions, events)
    metrics['c_index'] = c_index
    logging.info(f"C-index: {c_index}")

    # Calculate Brier Score
    brier_score = calculate_brier_score(durations, events, predictions)
    metrics['brier_score'] = brier_score
    logging.info(f"Brier Score: {brier_score}")

    return metrics, durations, events, predictions

def calculate_brier_score(durations : np.array, events : np.array, predictions : np.array):
    """
    Calculate the Brier score based on the durations, events and predictions

    Args:
        durations: The durations
        events: The events
        predictions: The predictions
    
    Returns:
        The Brier score

    """
    times = np.linspace(0, durations.max(), 100)
    brier_scores = []

    for t in times:
        brier_score = np.mean((predictions - (durations <= t) * events) ** 2)
        brier_scores.append(brier_score)

    return np.mean(brier_scores)

def plot_roc_curve_across_splits(rates : list, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the ROC curve across splits based on the rates

    Args:
        rates: The rates
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    fpr_list = [rate[1] for rate in rates]
    tpr_list = [rate[0] for rate in rates]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(len(fpr_list)):
        fpr = fpr_list[i]
        tpr = np.interp(mean_fpr, fpr, tpr_list[i])
        tpr[0] = 0.0
        tprs.append(tpr)
        roc_auc = auc(fpr_list[i], tpr_list[i])
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the ROC curve with shaded standard deviation area
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/roc_auc_{save_string}_{dataset}.png")

def plot_survival_auc_across_folds(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the survival ROC-AUC across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None

    """

    fig, ax = plt.subplots()
    all_aucs = []
    times_list = []

    for durations, events, predictions in results:
        # Ensure durations and events are numpy arrays
        durations = np.asarray(durations)
        events = np.asarray(events)
        
        # Create a structured array for survival data
        survival_data = np.array([(e, d) for e, d in zip(events, durations)], dtype=[('event', '?'), ('time', '<f8')])

        # Ensure time points are within the follow-up time range
        max_duration = durations.max()
        times = np.linspace(0, max_duration, 100)
        times_list.append(times)
        
        aucs = []
        for time in times:
            if time >= max_duration:
                continue  # Skip time points that are out of range

            # Calculate AUC and check for validity
            try:
                auc_score = cumulative_dynamic_auc(survival_data, survival_data, predictions, time)[0]
                if not np.isnan(auc_score):
                    aucs.append(auc_score)
            except ZeroDivisionError:
                continue

        if aucs:
            all_aucs.append(np.array(aucs).flatten())  # Ensure the AUC array is flattened
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            ax.plot(times[:len(aucs)], aucs, label=f'AUC (mean = {mean_auc:.2f}, std = {std_auc:.2f})')

    if all_aucs:
        # Determine the maximum length of the AUC lists
        max_len = max(len(auc) for auc in all_aucs)
        
        # Pad the aucs lists to have the same length
        padded_aucs = np.full((len(all_aucs), max_len), np.nan)
        for i, auc in enumerate(all_aucs):
            padded_aucs[i, :len(auc)] = auc

        mean_aucs = np.nanmean(padded_aucs, axis=0)
        std_aucs = np.nanstd(padded_aucs, axis=0)
        times = times_list[0]  # Use times from the first split
        ax.plot(times[:len(mean_aucs)], mean_aucs, label=f'Mean AUC (mean = {np.nanmean(mean_aucs):.2f}, std = {np.nanmean(std_aucs):.2f})', color='blue')

    ax.set_xlabel('Time')
    ax.set_ylabel('AUC')
    ax.set_title('Time-dependent ROC-AUC')
    ax.legend(loc="lower right")

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/survival_roc_auc_{save_string}_{dataset}.png")
    plt.close()


def ensure_monotonic(recall : list, precision : list):
    """
    Ensure that the recall is monotonically increasing and the precision is non-increasing

    Args:
        recall: The recall
        precision: The precision

    Returns:
        The recall and precision, ensuring the conditions are met
    """
    # Ensure that recall is monotonically increasing and precision is non-increasing
    for i in range(1, len(recall)):
        if recall[i] < recall[i - 1]:
            recall[i] = recall[i - 1]
    for i in range(1, len(precision)):
        if precision[i] > precision[i - 1]:
            precision[i] = precision[i - 1]
    return recall, precision

def plot_precision_recall_curve_across_splits(precision_recall_data : list, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the precision-recall curve across splits based on the precision-recall data

    Args:
        precision_recall_data: The precision-recall data, containing tuples of format (precision, recall)
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    aps = []
    
    all_y_true = np.concatenate([y_true for y_true, _ in precision_recall_data])
    baseline_precision = np.mean(all_y_true)  # Ratio of positive samples in the dataset

    for i, (y_true, y_pred) in enumerate(precision_recall_data):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        if recall.ndim > 1 or precision.ndim > 1:
            recall = recall.flatten()
            precision = precision.flatten()

        if len(recall) == 0 or len(precision) == 0:
            print(f"Empty precision or recall for split {i}")
            continue

        recall_reversed = recall[::-1]
        precision_reversed = precision[::-1]

        recall_reversed, precision_reversed = ensure_monotonic(recall_reversed, precision_reversed)

        try:
            interpolated_precision = np.interp(mean_recall, recall_reversed, precision_reversed)
            precisions.append(interpolated_precision)
            average_precision = auc(recall_reversed, precision_reversed)
            aps.append(average_precision)
            print(f"Interpolated precision for split {i}: {interpolated_precision}"
                  f"\nAverage precision for split {i}: {average_precision}")

        except ValueError as e:
            print(f"Error interpolating precision-recall for split {i}: {e}")
            continue

    if len(precisions) == 0:
        print("No valid precision-recall data available.")
        return

    mean_precision = np.nanmean(precisions, axis=0)
    std_precision = np.nanstd(precisions, axis=0)

    mean_ap = np.nanmean(aps)
    std_ap = np.nanstd(aps)

    # Plot the mean precision-recall curve with shaded standard deviation area
    plt.figure()
    plt.plot(mean_recall, mean_precision, color='b', label=r'Mean Precision-Recall (AP = %0.2f $\pm$ %0.2f)' % (mean_ap, std_ap), lw=2, alpha=.8)

    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # Add baseline precision-recall line
    plt.plot([0, 1], [baseline_precision, baseline_precision], linestyle='--', color='r', label='Baseline (random)')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/precision_recall_{save_string}_{dataset}.png")
    plt.close()

def plot_concordance_index_across_folds(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the concordance index across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    fig, ax = plt.subplots()
    times = np.linspace(0, max(durations.max() for durations, _, _ in results), 100)
    all_concordances = []

    for durations, events, predictions in results:
        fold_concordances = []
        for time in times:
            mask = durations >= time
            c_index = concordance_index(durations[mask], predictions[mask], events[mask])
            fold_concordances.append(c_index)
        all_concordances.append(fold_concordances)

    mean_concordances = np.mean(all_concordances, axis=0)
    std_concordances = np.std(all_concordances, axis=0)

    ax.plot(times, mean_concordances, marker='o', linewidth=1, label='Mean Concordance Index')
    ax.fill_between(times, mean_concordances - std_concordances, mean_concordances + std_concordances, alpha=0.2)

    ax.set_title('Concordance Index Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concordance Index')
    ax.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/concordance_index_{save_string}_{dataset}.png")
    plt.close()

def plot_predicted_vs_actual_across_folds(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the predicted vs actual values across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    all_y_true = []
    all_y_pred = []

    for y_true, y_pred in results:
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    mean_y_true = np.mean(all_y_true, axis=0)
    mean_y_pred = np.mean(all_y_pred, axis=0)
    std_y_pred = np.std(all_y_pred, axis=0)

    plt.figure()
    plt.errorbar(mean_y_true, mean_y_pred, yerr=std_y_pred, fmt='o', ecolor='r', capsize=2, label='Mean Predicted vs Actual')
    plt.plot([min(mean_y_true), max(mean_y_true)], [min(mean_y_true), max(mean_y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/predicted_vs_actual_{save_string}_{dataset}.png")
    plt.close()

def plot_calibration(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict, bins : int = 10):
    """
    Plot the calibration curve based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
        bins: The number of bins

    Returns:
        None
    """
    plt.figure()
    for durations, events, predictions in results:
        # Convert predicted log hazards to predicted risks
        predictions = np.exp(predictions)
        predictions = predictions / (1 + predictions)

        true_prob, pred_prob = calibration_curve(events, predictions, n_bins=bins, strategy='uniform')
        plt.plot(pred_prob, true_prob, marker='o', linewidth=1, label=f'{dataset}')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

def plot_residuals_across_folds(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the residuals across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """

    all_residuals = []

    for y_true, y_pred in results:
        residuals = y_true - y_pred
        all_residuals.append(residuals)

    all_residuals = np.array(all_residuals)
    mean_residuals = np.mean(all_residuals, axis=0)
    std_residuals = np.std(all_residuals, axis=0)

    plt.figure()
    plt.errorbar(np.arange(len(mean_residuals)), mean_residuals, yerr=std_residuals, fmt='o', ecolor='r', capsize=2, label='Mean Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/residuals_{save_string}_{dataset}.png")
    plt.close()

def plot_qq_across_folds(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Plot the Q-Q plot across folds based on the results
    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None
    """
    plt.figure()
    for i, (y_true, y_pred) in enumerate(results):
        residuals = y_true - y_pred
        stats.probplot(residuals, dist="norm", plot=plt)
    
    plt.title('Q-Q Plot')
    
    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/qq_{save_string}_{dataset}.png")
    plt.close()

def plot_calibration_across_splits(results : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict, bins : int = 10):
    """
    Plot the calibration curve across splits based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
        bins: The number of bins

    Returns:
        None
    """
    mean_pred_probs = []
    mean_true_probs = []
    all_pred_probs = []
    all_true_probs = []
    
    for durations, events, predictions in results:
        # Convert predicted log hazards to predicted risks
        predictions = np.exp(predictions)
        predictions = predictions / (1 + predictions)

        true_prob, pred_prob = calibration_curve(events, predictions, n_bins=bins, strategy='uniform')
        mean_pred_probs.append(pred_prob)
        mean_true_probs.append(true_prob)
        all_pred_probs.append(pred_prob)
        all_true_probs.append(true_prob)

    # Calculate the mean and standard error across splits
    mean_pred_probs = np.mean(mean_pred_probs, axis=0)
    mean_true_probs = np.mean(mean_true_probs, axis=0)
    std_true_probs = np.std(all_true_probs, axis=0)
    
    plt.figure()
    plt.plot(mean_pred_probs, mean_true_probs, marker='o', linewidth=1, label='Mean Calibration')
    plt.fill_between(mean_pred_probs, mean_true_probs - std_true_probs, mean_true_probs + std_true_probs, alpha=0.2, label='±1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

def calculate_uncertainty(result : pd.DataFrame, save_string : str, dataset : sf.Dataset, config : dict):
    """
    Calculate the uncertainty based on the results

    Args:
        result: The results dataframe
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    os.makedirs(f"experiments/{config['experiment']['project_name']}/results", exist_ok=True)
    result['uncertainty'] = result.apply(calculate_entropy, axis=1)
    result['certainty'] = 1 - result['uncertainty']
    result['group'] = result['certainty'].apply(assign_group)
    result.to_csv(f"experiments/{config['experiment']['project_name']}/results/{dataset}_results_{save_string}.csv")
    logging.info(f"Uncertainty calculated for {dataset} dataset and saved to experiments/{config['experiment']['project_name']}/results/{dataset}_results_{save_string}.csv")

def calculate_entropy(row : pd.Series):
    """
    Calculate the entropy based on the row (instance prediction).
    We calculate entropy as:
    -p * log2(p) - (1 - p) * log2(1 - p)
    where p is the probability of the positive class

    Args:
        row: The row of the prediction dataframe in question
    
    Returns:
        The entropy
    """
    p0 = row['y_pred0']
    p1 = row['y_pred1']
    # Ensure the probabilities are valid to avoid log(0)
    if p0 > 0 and p1 > 0:
        return - (p0 * np.log2(p0) + p1 * np.log2(p1))
    else:
        return 0.0

def assign_group(certainty : float):
    """
    Assign a group based on the certainty

    Args:
        certainty: The certainty value

    Returns:
        The group
    """
    if certainty <= 0.25:
        return '0-25'
    elif certainty <= 0.50:
        return '25-50'
    elif certainty <= 0.75:
        return '50-75'
    else:
        return '75-100'

"""
def get_top5_annotation_tiles_per_class(attention_map, bag_index, tile_directory, slide_name, diagnosis):
    #Get the top 5 attention values
    top5_attention_values = np.argsort(attention_map)[::-1][:5]
    print(f"Top 5 attention indices: {top5_attention_values}")
    top5_least_attended = np.argsort(attention_map)[:5]

    #Get the corresponding tile coordinates from bag_index
    top5_tile_coordinates = bag_index[top5_attention_values]
    print(f"Top 5 tile coordinates: {top5_tile_coordinates}")
    top5_least_attended_coordinates = bag_index[top5_least_attended]

    #Get the corresponding tile paths
    top5_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_tile_coordinates]
    top5_least_attended_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_least_attended_coordinates]
    print(f"Top 5 tile paths: {top5_tiles}")

    #Construct top 5 tiles directory
    os.makedirs(f"top5_tiles", exist_ok=True)

    #Plot the top 5 tiles using matplotlib
    import matplotlib.pyplot as plt
    import cv2
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, tile in enumerate(top5_tiles):
        img = Image.open(tile)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        axs[i].imshow(img)
        axs[i].axis('off')
    #Set global title above the subplots
    plt.suptitle(f"Top 5 most attended tiles ({diagnosis} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"top5_tiles/{diagnosis}_{slide_name}_top5_tiles.png")
    plt.close()
"""

def optimize_parameters(config, project):
    def objective(trial):
        # Load hyperparameters from the config file
        epochs = trial.suggest_int('epochs', config['experiment']['epochs'], config['experiment']['epochs'])
        batch_size = trial.suggest_int('batch_size', config['experiment']['batch_size'], config['experiment']['batch_size'])
        balancing = trial.suggest_categorical('balancing', [config['experiment']['balancing']])
        tile_px = trial.suggest_categorical('tile_px', config['benchmark_parameters']['tile_px'])
        tile_um = trial.suggest_categorical('tile_um', config['benchmark_parameters']['tile_um'])
        normalization = trial.suggest_categorical('normalization', config['benchmark_parameters']['normalization'])
        feature_extraction = trial.suggest_categorical('feature_extraction', config['benchmark_parameters']['feature_extraction'])
        mil = trial.suggest_categorical('mil', config['benchmark_parameters']['mil'])

        logging.info(f"Using suggested hyperparameters: epochs={epochs}, batch_size={batch_size}, balancing={balancing}, tile_px={tile_px}, tile_um={tile_um}, normalization={normalization}, feature_extraction={feature_extraction}, mil={mil}")

        # Update config with suggested hyperparameters
        config['experiment']['epochs'] = epochs
        config['experiment']['batch_size'] = batch_size
        config['experiment']['balancing'] = balancing

    
    sampler_class = getattr(optuna.samplers, config['optimization']['sampler'])
    pruner_class = getattr(optuna.pruners, config['optimization']['pruner'])

    study = optuna.create_study(sampler=optuna.samplers.config['optimization']['sampler']())

    # Create the study with the specified sampler and pruner
    study = optuna.create_study(sampler=sampler_class())

    # Optimize the study
    study.optimize(objective, n_trials=config['optimization']['trials'],
                pruner=pruner_class())