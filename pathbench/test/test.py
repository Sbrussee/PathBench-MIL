from pathbench.experiment.experiment import Experiment
import logging
import os



def test():
    root = "pathbench/test/"
    experiments_dir = "experiments/TCGA_LUSC"
    experiments_to_run = ['survival', 'classification', 'survival_discrete']
    #experiments_to_run = ['binary', 'classification', 'survival', 'discrete_survival', 'regression', 'opt']

    for exp in experiments_to_run:
        #Test binary classification
        if exp == 'binary':
            experiment = Experiment(root+'binary_test_conf.yaml')
            experiment.run()
            logging.info("Binary classification test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp1_binary_val_results_agg.csv and experiments/TCGA_LUSC/exp1_binary_test_results_agg.csv
            os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp1_binary_val_results_agg.csv")
            os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp1_binary_test_results_agg.csv")

        #Test multi-class classification
        if exp == 'classification':
            experiment = Experiment(root+'classification_test_conf.yaml')
            experiment.run()
            logging.info("Multiclass classification test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp2_classification_val_results_agg.csv and experiments/TCGA_LUSC/exp2_classification_test_results_agg.csv
            #os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp2_classification_val_results_agg.csv")
            #os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp2_classification_test_results_agg.csv")
        #Test survival prediction

        if exp == 'survival':
            experiment = Experiment(root+'survival_test_conf.yaml')
            experiment.run()
            logging.info("Survival prediction test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp3_survival_val_results_agg.csv and experiments/TCGA_LUSC/exp3_survival_test_results_agg.csv
            #os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp3_survival_val_results_agg.csv")
            #os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp3_survival_test_results_agg.csv")

        #Test discrete survival prediction
        if exp == 'survival_discrete':
            experiment = Experiment(root+'survival_discrete_test_conf.yaml')
            experiment.run()
            logging.info("Discrete survival prediction test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp3_survival_val_results_agg.csv and experiments/TCGA_LUSC/exp3_survival_test_results_agg.csv
            #os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp4_survival_discrete_val_results_agg.csv")
            #os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp4_survival_discrete_test_results_agg.csv")

        #Test regression
        if exp == 'regression':
            experiment = Experiment(root+'regression_test_conf.yaml')
            experiment.run()
            logging.info("Regression test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp4_regression_val_results_agg.csv and experiments/TCGA_LUSC/exp4_regression_test_results_agg.csv
            #os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp5_regression_val_results_agg.csv")
            #os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp5_regression_test_results_agg.csv")

        #Test optimization
        if exp == 'opt':
            experiment = Experiment(root+'opt_test_conf.yaml')
            experiment.run()
            logging.info("Optimization test passed")
            #Rename experiments/TCGA_LUSC/val_results_agg.csv and experiments/TCGA_LUSC/test_results_agg.csv to experiments/TCGA_LUSC/exp5_opt_val_results_agg.csv and experiments/TCGA_LUSC/exp5_opt_test_results_agg.csv
            os.rename(f"{experiments_dir}/results/val_results_agg.csv", f"{experiments_dir}/exp6_opt_val_results_agg.csv")
            os.rename(f"{experiments_dir}/results/test_results_agg.csv", f"{experiments_dir}/exp6_opt_test_results_agg.csv")

if __name__ == "__main__":
    test()