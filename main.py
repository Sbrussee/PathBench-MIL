from pathbench.experiment import Experiment
import argparse
import logging

# Create or get the root logger
logger = logging.getLogger(__name__)  # Use __name__ to get the module's logger
logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG to capture everything

# Formatter for both console and file
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 1. Console Handler (Logs INFO and higher)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only log INFO and higher to the console
console_handler.setFormatter(formatter)

# 2. File Handler (Logs DEBUG and higher)
file_handler = logging.FileHandler('debug_logfile.log')
file_handler.setLevel(logging.DEBUG)  # Log DEBUG and higher to the file
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Add arguments
parser = argparse.ArgumentParser(description='Run PathDev experiment')
parser.add_argument('--config', type=str, default='conf.yaml', help='Path to the configuration file')
args = parser.parse_args()


def main():
    """"
    Main function to run the experiment
    """
    # Create an instance of the Experiment class
    experiment = Experiment(args.config)
    experiment.run()

if __name__ == "__main__":
    main()