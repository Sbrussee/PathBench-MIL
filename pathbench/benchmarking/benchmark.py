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
    balanced_accuracy_score, f1_score, precision_recall_curve, recall_score, average_precision_score, confusion_matrix, roc_curve,
    auc, mean_absolute_error, mean_squared_error, r2_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
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

from ..utils.utils import *
from ..visualization.visualization import *



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
    finished_file_path = f"experiments/{config['experiment']['project_name']}/finished_combinations.pkl"
    
    directory = os.path.dirname(finished_file_path)
    os.makedirs(directory, exist_ok=True)

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
    
    logging.info("Starting benchmarking...")
    task = config['experiment']['task']
    benchmark_parameters = config['benchmark_parameters']
    project_directory = f"experiments/{config['experiment']['project_name']}"
    annotations = project.annotations
    logging.info(f"Using {project_directory} for project directory...")
    logging.info(f"Using {annotations} for annotations...")
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
                                    qc="both")
                                
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
                    metrics, durations, events, predictions = calculate_survival_results(val_result, config, save_string, "val")
                    survival_results_per_split.append((durations, events, predictions))
                elif config['experiment']['task'] == 'regression':
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
                    y_true, y_pred = val_result['y_true'], val_result['y_pred0']
                    val_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, save_string, "val")
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
                    metrics, durations, events, predictions = calculate_survival_results(test_result, config, save_string, "test")
                    test_survival_results_per_split.append((durations, events, predictions))
                elif config['experiment']['task'] == 'regression':
                    metrics, tpr, fprr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
                    y_true, y_pred = test_result['y_true'], test_result['y_pred0']
                    test_results_per_split.append((y_true, y_pred))
                else:
                    metrics, tpr, fpr, test_pr_curves = calculate_results(test_result, config, save_string, "test")
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


def calculate_results(result: pd.DataFrame, config: dict, save_string: str, dataset_type : str):
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
        all_y_true = result['y_true'].values
        all_y_pred_prob = []
        unique_classes = np.unique(result.y_true.values)

        for col in y_pred_cols:
            y_pred_prob = result[col].values
            class_label = int(col[-1])  # Get the class label from the column name

            all_y_pred_prob.append(y_pred_prob)

            # Binary predictions based on the optimal threshold
            m = ClassifierMetrics(y_true=(all_y_true == class_label).astype(int), y_pred=y_pred_prob)
            fpr, tpr, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = threshold[optimal_idx]
            y_pred_binary_opt = (y_pred_prob > optimal_threshold).astype(int)

            balanced_accuracy = balanced_accuracy_score((all_y_true == class_label).astype(int), y_pred_binary_opt)
            f1 = f1_score((all_y_true == class_label).astype(int), y_pred_binary_opt)

            balanced_accuracies.append(balanced_accuracy)
            aucs.append(auroc)
            average_precisions.append(average_precision_score((all_y_true == class_label).astype(int), y_pred_prob))
            average_recalls.append(recall_score((all_y_true == class_label).astype(int), y_pred_binary_opt))
            f1_scores[col] = f1

        # Compute overall class predictions
        all_y_pred_prob = np.vstack(all_y_pred_prob).T
        all_y_pred_class = np.argmax(all_y_pred_prob, axis=1)

        # Compute overall confusion matrix
        all_cm = confusion_matrix(all_y_true, all_y_pred_class, labels=unique_classes)

        # Plot overall confusion matrix
        save_path = f"experiments/{config['experiment']['project_name']}/visualizations"
        os.makedirs(save_path, exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=all_cm, display_labels=unique_classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Overall Confusion Matrix')
        plt.savefig(f"{save_path}/confusion_matrix_{save_string}_overall.png")
        plt.close()

        # Plot precision-recall curve
        y_true_binary = np.isin(all_y_true, unique_classes[unique_classes != unique_classes[-1]]).astype(int)
        y_pred_prob_binary = all_y_pred_prob[:, unique_classes != unique_classes[-1]].max(axis=1)
        plot_precision_recall_curve(y_true_binary, y_pred_prob_binary, save_path, save_string)

        # Store metrics
        metrics['balanced_accuracy'] = np.mean(balanced_accuracies)
        metrics['mean_f1'] = np.mean(list(f1_scores.values()))
        metrics['auc'] = np.mean(aucs)
        metrics['mean_average_precision'] = np.mean(average_precisions)
        metrics['mean_average_recall'] = np.mean(average_recalls)

        logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
        logging.info(f"Mean F1 Score: {metrics['mean_f1']}")
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
    study = optuna.create_study(sampler=sampler_class(), pruner=pruner_class())

    # Optimize the study
    study.optimize(objective, n_trials=config['optimization']['trials'])