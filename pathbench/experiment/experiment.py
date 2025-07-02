import yaml
from typing import List, Dict, Any
import slideflow as sf
import os
import torch
from ..benchmarking.benchmark import benchmark, optimize_parameters, extract_features
import random
import shutil
import logging
from huggingface_hub import login
import multiprocessing as mp
#import torch.multiprocessing as torch_mp
#torch_mp.set_sharing_strategy('file_system')

def read_config(config_file : str):
    """
    Read the configuration file for the experiment

    Args:
        config_file (str): The path to the configuration file
    
    Returns:
        dict: The configuration dictionary for the experiment
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

# Check if GPU is available
if torch.cuda.is_available():
    try:
        # Check if device already set
        device = torch.cuda.current_device()
    except NameError:
        # Set device to GPU
        device = torch.device('cuda')
        torch.cuda.set_device(device)
        logging.info(f'Using GPU: {torch.cuda.get_device_name(device)}')
else:
    try:
        # Check if device already set
        device = torch.cuda.current_device()
    except NameError:
        # Set device to CPU
        device = torch.device('cpu')
        logging.info('Using CPU')
        
class Experiment():
    """
    Experiment class, designed to load the data and the configuration of a benchmarking
    experiement and run the benchmarking experiment.

    Parameters
    ----------
    config_file : str
        The path to the configuration file for the experiment
    
    Attributes
    ----------
    config : dict
        The configuration dictionary for the experiment
    project : slideflow.Project
        The slideflow project object
    project_name : str
        The name of the project
    
    Methods
    -------
    run()
        Run the benchmarking experiment
    load_datasets()
        Load the datasets into the project
    benchmark()
        Run the benchmarking experiment
    
    """
    def __init__(self, config_file : str):
        self.config = read_config(config_file)
        logging.info(f"Configuration file {config_file} loaded")
        self.load_datasets()
        #Set Hugging Face token
        if 'hf_key' in self.config:
            if self.config['hf_key'] is not None:
                HF_TOKEN = self.config['hf_key']
                os.environ['HF_TOKEN'] = HF_TOKEN
                if HF_TOKEN == 'None':
                    logging.info("No Hugging Face token provided")
                else:
                    login(token=HF_TOKEN)
                    logging.info("Logged in to Hugging Face")

        #Set pretrained weights directory
        WEIGHTS_DIR = self.config['weights_dir']
        # Set environment variables
        os.environ['TORCH_HOME'] = WEIGHTS_DIR
        os.environ['HF_HOME'] = WEIGHTS_DIR
        os.environ['XDG_CACHE_HOME'] = WEIGHTS_DIR
        os.environ['HF_DATASETS_CACHE'] = WEIGHTS_DIR
        os.environ['WEIGHTS_DIR'] = WEIGHTS_DIR

        logging.info(f"Set environment variables for pretrained weights directory: {WEIGHTS_DIR}")

    def run(self):
        if self.config['experiment']['mode'] == 'benchmark':
            logging.info("Running benchmarking mode...")
            self.benchmark()
        elif self.config['experiment']['mode'] == 'optimization':
            logging.info("Running optimization mode...")
            self.optimize_parameters()
        elif self.config['experiment']['mode'] == "feature_extraction":
            logging.info("Running feature extraction mode...")
            self.extract_features()
        else:
            raise ValueError("Invalid mode. Mode must be either 'benchmark' or 'optimization'")

    def load_datasets(self):
        """
        Load datasets into the project. the datasets are specified in the configuration file.
        We assume that the first dataset is the main dataset and the rest are additional datasets.
        As such, we create a project based on the first dataset and add the rest of the datasets
        as sources to the project.
        """
        # Ensure experiments directory exists
        os.makedirs('experiments', exist_ok=True)

        # Project setup
        self.project_name = self.config['experiment']['project_name']
        project_path = f"experiments/{self.project_name}"
        annotation_file = self.config['experiment']['annotation_file']

        # Load existing project or create a new empty one
        if os.path.exists(project_path):
            logging.info(f"Loading project {self.project_name}")
            self.project = sf.Project(
                project_path,
                annotations=annotation_file
            )
            logging.info(f"Project {self.project_name} loaded with annotations: {self.project.annotations}")
        else:
            logging.info(f"Creating new project {self.project_name}")
            os.makedirs(project_path, exist_ok=True)
            # Create an empty project (no initial slides/tiles/tfrecords)
            self.project = sf.create_project(
                name=self.project_name,
                root=project_path,
                annotations=annotation_file
            )
            logging.info(f"Project {self.project_name} created with annotations: {self.project.annotations}")

        # Retrieve datasets list and validate
        datasets = self.config.get('datasets', [])
        if not datasets:
            raise ValueError(f"No datasets specified in the configuration for project '{self.project_name}'.")

        # Add each dataset as a source to the project
        for source in datasets:

            name = source.get('name')
            if not name:
                raise ValueError(f"Dataset name is missing in the configuration for project '{self.project_name}'.")
            
            def resolve_path(path):
                if not path:
                    return None
                is_absolute = os.path.isabs(path)
                if is_absolute:
                    logging.debug(f"Using absolute path: {path}")
                    return path
                else:
                    path = os.path.join(os.getcwd(), path)
                    logging.debug(f"Interpreted as relative path: {path}")
                    if not os.path.exists(path):
                        raise ValueError(f"Path '{path}' does not exist.")
                    return path
            

            self.project.add_source(
                name=source['name'],
                slides=resolve_path(source.get('slide_path')),
                tfrecords=resolve_path(source.get('tfrecord_path')),
                tiles=resolve_path(source.get('tile_path')) if self.config['experiment'].get('save_tiles', None) else None,
            )
            logging.info(f"Added source '{source['name']}' to project '{self.project_name}'")

        
    def benchmark(self):
        #Iterate over all possible combinations of hyperparameters
        benchmark(self.config, self.project)

    def optimize_parameters(self):
        #Optimize the MIL pipeline
        optimize_parameters(self.config, self.project)

    def extract_features(self):
        #Extract features from the dataset
        extract_features(self.config, self.project)
