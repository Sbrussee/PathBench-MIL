from pathbench.experiment.experiment import Experiment
import logging



def main():
    #Test binary classification
    experiment = Experiment('binary_test_conf.yaml')
    experiment.run()
    logging.info("Binary classification test passed")
    #Test multi-class classification
    experiment = Experiment('classification_test_conf.yaml')
    experiment.run()
    logging.info("Multiclass classification test passed")
    #Test survival prediction
    experiment = Experiment('survival_test_conf.yaml')
    experiment.run()
    logging.info("Survival prediction test passed")
    #Test regression
    experiment = Experiment('regression_test_conf.yaml')
    experiment.run()
    logging.info("Regression test passed")
    #Test optimization
    experiment = Experiment('opt_test_conf.yaml')
    experiment.run()
    logging.info("Optimization test passed")

main()