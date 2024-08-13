import yaml
import slideflow as sf
import os
import torch
from ..benchmarking.benchmark import benchmark
from ..benchmarking.benchmark import optimize_parameters
import random
import shutil
import logging
from ..utils.utils import increase_file_limit

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
        #Set pretrained weights directory
        WEIGHTS_DIR = self.config['weights_dir']
        # Set environment variables
        os.environ['TORCH_HOME'] = WEIGHTS_DIR
        os.environ['HF_HOME'] = WEIGHTS_DIR

        #Increase open file limit, useful for multiprocessing
        increase_file_limit(16384)

    def run(self):
        if self.config['experiment']['mode'] == 'benchmark':
            logging.info("Running benchmarking mode...")
            self.benchmark()
        elif self.config['experiment']['mode'] == 'optimization':
            logging.info("Running optimization mode...")
            self.optimize_parameters()
        else:
            raise ValueError("Invalid mode. Mode must be either 'benchmark' or 'optimization'")

    def load_datasets(self):
        """
        Load datasets into the project
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
        else:
            logging.info(f"Creating project {self.project_name}")
            os.makedirs(f"experiments", exist_ok=True)
            os.makedirs(f"experiments/{self.project_name}", exist_ok=True)
            self.project = sf.create_project(
                name=self.project_name,
                root=f"experiments/{self.project_name}",
                annotations=self.config['experiment']['annotation_file'],
                slides=first_dataset['slide_path']
            ) 
            logging.info(f"Project {self.project_name} created")

        #Add additional datasets to the project
        if len(self.config['datasets']) > 1:
            for source in self.config['datasets'][1:]:
                self.project.add_source(
                    name=source['name'],
                    slides=source['slide_path'],
                    tfrecords=source['tfrecord_path'],
                    tiles=source['tile_path']
                )
        
        
    """
    def ssl(self):
        ssl_parameters = self.config['ssl']
        (method, backbone, train_path, val_path, val_split,
        ssl_model_name) = (ssl_parameters['method'],
            ssl_parameters['backbone'], ssl_parameters['train_path'],
            ssl_parameters['val_path'], ssl_parameters['val_split'],
            ssl_parameters['ssl_model_name'])
        
        os.makedirs(f'{self.config["experiment"]["project_name"]}/ssl_train', exist_ok=True)
        os.makedirs(f'{self.config["experiment"]["project_name"]}/ssl_val', exist_ok=True)

        if val_path == None and val_split != None:
            #Get all directories in the training directory
            train_slides = os.listdir(train_path)
            print("Total number of slides: ", len(train_slides))

            #Shuffle the directory
            random.shuffle(train_slides)
            #Split training data into training and validation data
            val_size = int(val_split * len(train_slides))
            val_files = train_slides[:val_size]
            train_files = train_slides[val_size:]

            for slide in train_files:
                images = os.listdir(f'{train_path}/{slide}')
                for image in images:
                    shutil.copy(f'{train_path}/{slide}/{image}', f"{self.config['experiment']['project_name']}/ssl_train/{image}")

            for slide in val_files:
                images = os.listdir(f'{train_path}/{slide}')
                for image in images:
                    shutil.copy(f'{train_path}/{slide}/{image}', f"{self.config['experiment']['project_name']}/ssl_val/{image}")

        elif val_path != None:
            #Copy all files from the training directory to the ssl_train directory
            for slide in os.listdir(train_path):
                for image in os.listdir(f"{train_path}/{slide}"):
                    shutil.copy(f'{train_path}/{slide}/{image}', f"{self.config['experiment']['project_name']}/ssl_train/{image}")
            #Copy all files from the validation directory to the ssl_val directory
            for slide in os.listdir(val_path):
                for image in os.listdir(f"{val_path}/{slide}"):
                    shutil.copy(f'{val_path}/{slide}/{image}', f"{self.config['experiment']['project_name']}/ssl_val/{image}")
        else:
            for slide in os.listdir(train_path):
                for image in os.listdir(f"{train_path}/{slide}"):
                    shutil.copy(f'{train_path}/{slide}/{image}', f"{self.config['experiment']['project_name']}/ssl_train/{image}")

        if val_path != None:
            train_ssl_model(method, backbone, ssl_model_name, f"{self.config['experiment']['project_name']}/ssl_train",
                        f"{self.config['experiment']['project_name']}/ssl_val")
        else:
            train_ssl_model(method, backbone, ssl_model_name, f"{self.config['experiment']['project_name']}/ssl_train")
    """
    def benchmark(self):
        #Iterate over all possible combinations of hyperparameters
        benchmark(self.config, self.project)

    def optimize_parameters(self):
        optimize_parameters(self.config, self.project)

    """
    def hpo(self):
        optimizer = HyperParameterOptimizer()
        optimizer.run()
    """
