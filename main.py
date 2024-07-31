from pathbench.experiment.experiment import Experiment

def main():
    """"
    Main function to run the experiment
    """
    # Create an instance of the Experiment class
    experiment = Experiment('conf.yaml')
    experiment.run()

if __name__ == "__main__":
    main()