from pathbench.experiment.experiment import Experiment


def main():
    experiment = Experiment('config.yaml')
    experiment.run()

if __name__ == "__main__":
    main()