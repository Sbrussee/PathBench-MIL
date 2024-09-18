import yaml
import slideflow as sf
import os
import torch
from ..benchmarking.benchmark import benchmark, optimize_parameters, ensemble
import random
import shutil
import logging
from huggingface_hub import login
import multiprocessing as mp
mp.set_start_method('fork')

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
    # Get the current GPU device
    device = torch.cuda.current_device()

    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(device)

    print(f'Using GPU: {gpu_name}')
else:
    print('GPU not available. Using CPU.')

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
        if 'hf_token' in self.config:
            if self.config['hf_token'] is not None:
                HF_TOKEN = self.config['hf_token']
                os.environ['HF_TOKEN'] = HF_TOKEN
                login(token=HF_TOKEN)
                logging.info("Logged in to Hugging Face")
        #Set pretrained weights directory
        WEIGHTS_DIR = self.config['weights_dir']
        # Set environment variables
        os.environ['TORCH_HOME'] = WEIGHTS_DIR
        os.environ['HF_HOME'] = WEIGHTS_DIR

    def run(self):
        if self.config['experiment']['mode'] == 'benchmark':
            logging.info("Running benchmarking mode...")
            self.benchmark()
        elif self.config['experiment']['mode'] == 'optimization':
            logging.info("Running optimization mode...")
            self.optimize_parameters()
        elif self.config['experiment']['mode'] == 'ensemble':
            logging.info("Running ensemble mode...")
            self.ensemble()
        else:
            raise ValueError("Invalid mode. Mode must be either 'benchmark' or 'optimization'")

    def load_datasets(self):
        """
        Load datasets into the project. the datasets are specified in the configuration file.
        We assume that the first dataset is the main dataset and the rest are additional datasets.
        As such, we create a project based on the first dataset and add the rest of the datasets
        as sources to the project.
        """
        #Create an experiment folder
        os.makedirs('experiments', exist_ok=True)

        self.project_name = self.config['experiment']['project_name']
        if 'datasets' in self.config:
            first_dataset = self.config['datasets'][0]
        #Create project based on first dataset
        #Check if project exists
        if os.path.exists(f"experiments/{self.project_name}"):
            logging.info(f"Loading project {self.project_name}")
            self.project = sf.Project(f"experiments/{self.project_name}",
            annotations=self.config['experiment']['annotation_file'])
            logging.info(f"Project {self.project_name} loaded")
            logging.info(f"Annotations in project: {self.project.annotations}")
            for index, source in enumerate(self.project.sources):
                logging.info(f"Source {index}: {source}")
                logging.info(f"Slides in source: {self.config['datasets'][index]['slide_path']}")

        else:
            logging.info(f"Creating project {self.project_name}")
            os.makedirs(f"experiments", exist_ok=True)
            os.makedirs(f"experiments/{self.project_name}", exist_ok=True)
            self.project = sf.create_project(
                name=self.project_name,
                root=f"experiments/{self.project_name}",
                annotations=self.config['experiment']['annotation_file'],
                slides=first_dataset['slide_path'])
            
            logging.info(f"Project {self.project_name} created")
            logging.info(f"Annotations in project: {self.project.annotations}")

        #Add additional datasets to the project
        if len(self.config['datasets']) > 1:
            for source in self.config['datasets'][1:]:
                self.project.add_source(
                    name=source['name'],
                    slides=source['slide_path'],
                    tfrecords=source['tfrecord_path'],
                    tiles=source['tile_path']
                )
                logging.info(f"Added source {source['name']} to project {self.project_name}")
                logging.info(f"Slides in source: {source['slide_path']}")
        
    def benchmark(self):
        #Iterate over all possible combinations of hyperparameters
        benchmark(self.config, self.project)

    def optimize_parameters(self):
        #Optimize the MIL pipeline
        optimize_parameters(self.config, self.project)

    def ensemble(self):
        #Build a model ensemble
        ensemble(self.project, self.config)