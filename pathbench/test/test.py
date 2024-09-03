from pathbench.experiment.experiment import Experiment
import logging



def test():
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
    #Test ensemble modeling
    experiment = Experiment(root+'ensemble_test_conf.yaml')
    experiment.run()
    logging.info("Ensemble modeling test passed")

if __name__ == "__main__":
    test()