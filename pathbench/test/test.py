from pathbench.experiment.experiment import Experiment
import logging



def main():
    root = "pathbench/test/"
    #Test binary classification
    experiment = Experiment(root+'binary_test_conf.yaml')
    experiment.run()
    logging.info("Binary classification test passed")
    #Test multi-class classification
    experiment = Experiment(root+'classification_test_conf.yaml')
    experiment.run()
    logging.info("Multiclass classification test passed")
    #Test survival prediction
    experiment = Experiment(root+'survival_test_conf.yaml')
    experiment.run()
    logging.info("Survival prediction test passed")
    #Test regression
    experiment = Experiment(root+'regression_test_conf.yaml')
    experiment.run()
    logging.info("Regression test passed")
    #Test optimization
    experiment = Experiment(root+'opt_test_conf.yaml')
    experiment.run()
    logging.info("Optimization test passed")

main()