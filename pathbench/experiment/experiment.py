import yaml
import slideflow as sf
import os
import torch
from ..optimization.hpo import HyperParameterOptimizer
from ..benchmarking.benchmark import benchmark
from ..models.ssl import train_ssl_model
import random
import shutil

def read_config(config_file):
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
    def __init__(self, config_file):
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

        first_dataset = self.config['datasets'][0]
        #Create project based on first dataset
        #Check if project exists
        if os.path.exists(f"experiments/{self.project_name}"):
            self.project = sf.load_project(f"experiments/{self.project_name}")
        else:
            self.project = sf.create_project(
                name=self.project_name,
                root=f"experiments/{self.project_name}",
                annotations=self.config['experiment']['annotation_file'],
                slides=first_dataset['slide_path']
            ) 
        #Add additional datasets to the project
        if len(self.config['datasets']) > 1:
            for source in self.config['datasets'][1:]:
                self.project.add_source(
                    name=source['name'],
                    slides=source['slide_path']
                )
        
    
    def ssl(self):
        ssl_parameters = self.config['ssl']
        (method, backbone, train_path, val_path, val_split,
        ssl_model_name) = (ssl_parameters['method'],
            ssl_parameters['backbone'], ssl_parameters['train_path'],
            ssl_parameters['val_path'], ssl_parameters['val_split'],
            ssl_parameters['ssl_model_name'])
        
        os.makedirs(f'{self.config['experiment']['project_name']}/ssl_train', exist_ok=True)
        os.makedirs(f'{self.config['experiment']['project_name']}/ssl_val', exist_ok=True)

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
                    shutil.copy(f'{train_path}/{slide}/{image}', f'{self.config['experiment']['project_name']}/ssl_train/{image}')

            for slide in val_files:
                images = os.listdir(f'{train_path}/{slide}')
                for image in images:
                    shutil.copy(f'{train_path}/{slide}/{image}', f'{self.config['experiment']['project_name']}/ssl_val/{image}')

        elif val_path != None:
            #Copy all files from the training directory to the ssl_train directory
            for slide in os.listdir(train_path):
                for image in os.listdir(slide):
                    shutil.copy(f'{train_path}/{slide}/{image}', f'{self.config['experiment']['project_name']}/ssl_train/{image}')
            #Copy all files from the validation directory to the ssl_val directory
            for slide in os.listdir(val_path):
                for image in os.listdir(slide):
                    shutil.copy(f'{val_path}/{slide}/{image}', f'{self.config['experiment']['project_name']}/ssl_val/{image}')
        else:
            for slide in os.listdir(train_path):
                for image in os.listdir(slide):
                    shutil.copy(f'{train_path}/{slide}/{image}', f'{self.config['experiment']['project_name']}/ssl_train/{image}')

        if val_path != None:
            train_ssl_model(method, backbone, ssl_model_name, f'{self.config['experiment']['project_name']}/ssl_train',
                        f'{self.config['experiment']['project_name']}/ssl_val')
        else:
            train_ssl_model(method, backbone, ssl_model_name, f'{self.config['experiment']['project_name']}/ssl_train')
    
    def benchmark(self):
        #Iterate over all possible combinations of hyperparameters
        benchmark(self.config, self.project)



    def hpo(self):
        optimizer = HyperParameterOptimizer()
        optimizer.run()

