from pathbench.experiment.experiment import Experiment
import logging
import time

def test():
    root = "pathbench/test/"
    experiments_to_run = ['opt', 'opt_naive']
    #experiments_to_run = ['binary', 'classification', 'survival', 'survival_discrete', 'regression', 'opt']

    for exp in experiments_to_run:
        #Test binary classification
        if exp == 'binary':
            experiment = Experiment(root+'binary_test_conf.yaml')
            experiment.run()
            logging.info("Binary classification test passed")

        #Test multi-class classification
        if exp == 'classification':
            experiment = Experiment(root+'classification_test_conf.yaml')
            experiment.run()
            logging.info("Multiclass classification test passed")

        if exp == 'survival':
            experiment = Experiment(root+'survival_test_conf.yaml')
            experiment.run()
            logging.info("Survival prediction test passed")
        #Test discrete survival prediction
        if exp == 'survival_discrete':
            experiment = Experiment(root+'survival_discrete_test_conf.yaml')
            experiment.run()
            logging.info("Discrete survival prediction test passed")

        #Test regression
        if exp == 'regression':
            experiment = Experiment(root+'regression_test_conf.yaml')
            experiment.run()
            logging.info("Regression test passed")
        #Test optimization
        if exp == 'opt':
            #Start time measurement
            start_time = time.time()

            experiment = Experiment(root+'opt_test_conf.yaml')
            experiment.run()

            #End time measurement
            end_time = time.time()
            elapsed_time = end_time - start_time
            #Save time in minutes to a file
            # Create results directory if it doesn't exist
            os.makedirs(f"{root}/experiments/TCGA_LUSC/results/", exist_ok=True)
            with open(f"{root}/experiments/TCGA_LUSC/results/opt_test_time.txt", "w") as f:
                f.write(f"Optimization test time: {elapsed_time/60} minutes\n")
            logging.info("Optimization test passed")

        if exp == "opt_naive":
            #Start time measurement
            start_time = time.time()

            experiment = Experiment(root+'opt_naive_test_conf.yaml')
            experiment.run()

            #End time measurement
            end_time = time.time()
            elapsed_time = end_time - start_time
            #Save time in minutes to a file
            with open(f"{root}experiments/TCGA_LUSC/results/opt_naive_test_time.txt", "w") as f:
                f.write(f"Optimization naive test time: {elapsed_time/60} minutes\n")
            logging.info("Optimization naive test passed")
            
if __name__ == "__main__":
    test()