import pandas as pd
import numpy as np
import os
import logging
import resource
import gc
import multiprocessing
import weakref
import psutil
import torch

CURRENT_OPEN_FILES = []

def remove_cache():
    """
    Remove the cache directories
    """
    #Remove /home/user/.cache/huggingface
    default_cache = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(default_cache):
        os.system(f"rm -rf {default_cache}")
    #Remove /home/user/.cache/torch
    default_cache = os.path.expanduser("~/.cache/torch")
    if os.path.exists(default_cache):
        os.system(f"rm -rf {default_cache}")
    return
    
def get_available_gpus():
    """
    Retrieves the number of available GPUs, prioritizing SLURM environment variables
    if running in a SLURM job. Falls back to PyTorch if SLURM variables are not found.

    Returns:
        int: Number of GPUs available.
    """
    # Check SLURM variables
    slurm_gpus = os.getenv("SLURM_GPUS_ON_NODE")
    if slurm_gpus:
        return int(slurm_gpus)

    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    if slurm_job_gpus:
        return len(slurm_job_gpus.split(","))

    # Fallback to PyTorch
    return torch.cuda.device_count()
    
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

def list_open_files():
    """
    List the open files in the process

    Returns:
        The list of open files
    """
    # Get the process ID
    pid = os.getpid()

    #Get process
    process = psutil.Process(pid)

    #Get open files
    open_files = process.open_files()

    #Print all newly opened files (not in current open files)
    for file in open_files:
        logging.debug("Number of open files:", len(open_files))
        if file not in CURRENT_OPEN_FILES:
            CURRENT_OPEN_FILES.append(file)
            logging.debug("Newly opened file:", file.path)
            logging.debug("File descriptor:", file.fd)
            logging.debug("File mode:", file.mode)

    

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

def get_mil_directory_number(source_directory : str, save_string: str):
    """
    Get the directory number for the MIL experiment based on the source directory and save string.

    Args:
        source_directory: The source directory of the experiment
        save_string: The save string to identify the experiment

    Returns:
        The directory number as an integer
    """
    files = os.listdir(source_directory)

    # Select the files that contain the save_string
    relevant_files = [f for f in files if save_string in f]
    #If multiple files are found, select the one with the highest number
    if relevant_files:
        # Extract the numbers from the filenames
        numbers = [int(f.split('-')[0]) for f in relevant_files if f.split('-')[0].isdigit()]
        # Return the maximum number found
        number = max(numbers)
        #Append 0's to front of number to make it 5-digit long
        return str(number).zfill(5)
    else:
        # If no relevant files are found, return 0
        return 0

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

def save_correct(result: pd.DataFrame, save_string: str, dataset_type: str, config: dict):
    """
    Save CSV files with correct and incorrect predictions.

    Args:
        result: The results dataframe
        save_string: The save string
        dataset_type: Type of dataset (e.g., "val" or "test")
        config: The configuration dictionary

    Returns:
        None
    """
    # Identify correct predictions
    y_true = result['y_true'].values
    y_pred_cols = [c for c in result.columns if c.startswith('y_pred')]
    
    # For multiclass classification, find the column with the highest probability for each prediction
    y_pred_probs = result[y_pred_cols].values
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Add a column indicating whether the prediction is correct
    result['correct'] = (y_true == y_pred).astype(int)
    
    # Filter correct and incorrect predictions
    correct_predictions = result[result['correct'] == 1]
    incorrect_predictions = result[result['correct'] == 0]

    # Save correct predictions to a CSV file
    correct_save_path = f"experiments/{config['experiment']['project_name']}/results/{save_string}_{dataset_type}_correct.csv"
    incorrect_save_path = f"experiments/{config['experiment']['project_name']}/results/{save_string}_{dataset_type}_incorrect.csv"
    os.makedirs(os.path.dirname(correct_save_path), exist_ok=True)
    
    correct_predictions.to_csv(correct_save_path, index=False)
    incorrect_predictions.to_csv(incorrect_save_path, index=False)

    logging.info(f"Correct predictions saved to {correct_save_path}")
    logging.info(f"Incorrect predictions saved to {incorrect_save_path}")


def free_up_gpu_memory():
    """
    Function to free up GPU memory in PyTorch.
    """
    # Free up cached memory in PyTorch
    torch.cuda.empty_cache()
    
    # Set the environment variable to reduce memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Optional: Print confirmation that the memory has been freed and the setting applied
    logging.debug("GPU memory has been cleared and expandable_segments has been enabled.")