from pathbench.experiment.experiment import Experiment
import argparse

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