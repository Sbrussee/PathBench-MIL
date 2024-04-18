import yaml
import slideflow as sf
import os
import torch
from ..optimization.hpo import HyperParameterOptimizer
from ..benchmarking.benchmark import benchmark
from ..models.ssl import train_ssl_model

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)
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
    def __init__(self, config_file, mode):
        self.config = read_config(config_file)
        self.mode = self.config['mode']
        self.load_datasets()

    def run(self):
        if self.mode == 'exploration':
            self.benchmark()
        elif self.mode == 'hpo':
            self.hpo()
        elif self.mode == 'ssl':
            self.ssl()

    def load_datasets(self):
        #Create an experiment folder
        os.makedirs('experiments', exist_ok=True)

        self.project_name = self.config['experiment']['project_name']

        first_dataset = self.config['dataset'][0]
        #Create project based on first dataset
        self.project = sf.create_project(
            name=self.project_name,
            root=f"experiments/{self.project_name}",
            annotations=first_dataset['annotation_path'],
            slides=first_dataset['slide_path']
        ) 
        #Add additional datasets to the project
        if len(self.config['datasets'] > 1):
            for source in self.config['datasets'][1:]:
                self.project.add_source(
                    name=source['name'],
                    slides=source['slide_path']
                )
        
    
    def ssl(self):
        ssl_parameters = self.config['ssl']
        (method, backbone, train_path, val_path,
        ssl_model_name) = (ssl_parameters['method'],
            ssl_parameters['backbone'], ssl_parameters['train_path'],
            ssl_parameters['val_path'], ssl_parameters['ssl_model_name'])
        train_ssl_model(method, backbone, ssl_model_name, train_path, val_path)
    
    def benchmark(self):
        #Iterate over all possible combinations of hyperparameters
        benchmark(self.config, self.project)



    def hpo(self):
        optimizer = HyperParameterOptimizer()
        optimizer.run()

