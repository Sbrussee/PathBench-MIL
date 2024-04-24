from itertools import product
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import slideflow as sf
from slideflow.model import build_feature_extractor
from slideflow.stats.metrics import ClassifierMetrics
from slideflow.mil import mil_config
from ..visualization.visualization import visualize_activations
"""


combination_dict shape:
{
    'normalization': 'macenko',
    'feature_extraction': 'CTransPath',
    'mil': 'CLAM_SB'
}

"""
def benchmark(config, project):
    #Get all column values
    columns = list(config['benchmark_parameters'].keys())
    columns.extend(list(config['experiment']['evaluation']))                   
    val_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    # Retrieve all combinations
    benchmark_parameters = config['benchmark_parameters']
    combinations = []
    for values in benchmark_parameters.values():
        if isinstance(values, list):
            combinations.append(values)

    all_combinations = list(product(*combinations))

    # Iterate over combinations
    for combination in all_combinations:
        combination_dict = {}
        for parameter_name, parameter_value in zip(benchmark_parameters.keys(), combination):
            combination_dict[parameter_name] = parameter_value
        
        save_string = "_".join([f"{value}" for value in combination_dict.values()])
        #Run with current parameters

        #Split datasets into train, val and test
        all_data = project.dataset(tile_px=combination_dict['tile_px'],
                                   tile_um=combination_dict['tile_um'],
                                   )
        
                #Extract tiles with QC for all datasets
        all_data.extract_tiles(enable_downsample=False,
                              save_tiles=True,
                              img_format="png",
                              qc="both")
        
        feature_extractor = build_feature_extractor(combination_dict['feature_extraction'],
                                                    tile_px=combination_dict['tile_px'])
        
        #Generate feature bags.
        os.makedirs(f"experiments/{config['experiment']['project_name']}/bags", exist_ok=True)
        bags = project.generate_feature_bags(model=feature_extractor, 
                                             dataset= all_data,
                                             normalizer=combination_dict['normalization'],
                                             outdir=f"experiments/{config['experiment']['project_name']}/bags/{save_string}")
        #Currently gives OUTOFMEMORY error, needs reworking
        """
        if 'layer_activations' in config['experiment']['visualization']:
            features = sf.DatasetFeatures(model=feature_extractor,
                                          dataset=all_data,
                                          normalizer=combination_dict['normalization'])
            visualize_activations(features, config, all_data, save_string)
        """
        train_set = all_data.filter(filters={'dataset' : 'train'})
        
        try:
            train_set.balance(headers='category', strategy=config['experiment']['balancing'])
        except:
            print("Train set balancing failed.")
        test_set = all_data.filter(filters={'dataset' : 'validate'})

        if config['experiment']['split_technique'] == 'k-fold':
            k = config['experiment']['k']

            splits = train_set.kfold_split(k=k, labels='category')
        else:
            splits = train_set.split(labels='category',
                                     val_strategy=config['experiment']['split_technqiue'],
                                     val_fraction=config['experiment']['val_split'])
        
        #Set MIL configuration
        config= mil_config(combination_dict['mil'].lower(), aggregation_level=config['experiment']['aggregation_level'])

        index = 1
        
        for train, val in splits:
            val_result = project.train_mil(
                config=config,
                outcomes='category',
                train_dataset=train,
                val_dataset=val,
                bags=bags,
                exp_label=f"{save_string}_{index}"
            )
            metrics = calculate_results(val_result, config, save_string)
            val_dict = combination_dict.copy()
            val_dict.update(metrics)

            val_df = val_df.append(val_dict, ignore_index=True)

            test_result = project.evaluate_mil(
                model = f"experiments/{config['experiment']['project_name']}/mil/{save_string}_{index}",
                outcomes='category',
                dataset=test_set,
                bags=f"experiments/{config['experiment']['project_name']}/bags/{save_string}",
                config=config
            )
            
            metrics = calculate_results(test_result, config, save_string)
            test_dict = combination_dict.copy()
            test_dict.update(metrics)
            test_df = test_df.append(test_dict, ignore_index=True)

            index += 1
        print(f"Combination {save_string} finished...")
            
    #Group dataframe and save
    val_df = val_df.groupby(list(benchmark_parameters.keys()))
    test_df = test_df.groupby(list(benchmark_parameters.keys()))

    val_df_agg = val_df.agg(config['experiment']['evaluation'])
    test_df_agg = test_df.agg(config['experiment']['evaluation'])

    #Save all dataframes
    os.makedirs(f"{config['experiment']['name']}/results", exist_ok=True)
    val_df.to_csv(f"{config['experiment']['name']}/results/val_results_{save_string}.csv")
    test_df.to_csv(f"{config['experiment']['name']}/results/test_results_{save_string}.csv")
    val_df_agg.to_csv(f"{config['experiment']['name']}/results/val_results_agg_{save_string}.csv")
    test_df_agg.to_csv(f"{config['experiment']['name']}/results/test_results_agg_{save_string}.csv")






def calculate_results(result, config, save_string):
    metrics = {}
    print(result)
    y_pred_cols = [c for c in result.columns if c.startswith('y_pred')]
    for idx in range(len(y_pred_cols)):
        m = ClassifierMetrics(
            y_true=(result.y_true.values == idx).astype(int),
            y_pred=result[f'y_pred{idx}'].values
        )

        fpr, tpr, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = threshold[optimal_idx]
        y_pred_binary = (result[f'y_pred{idx}'].values > optimal_threshold).astype(int)

        balanced_accuracy = balanced_accuracy_score((result.y_true.values == idx).astype(int), y_pred_binary)
        print(f"BA cat #{idx}: {balanced_accuracy}")
        metrics['balanced_accuracy'] = balanced_accuracy
        metrics['auc'] = auroc


    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="orange",
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % auroc,
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    os.makedirs(f"{config['experiment']['name']}/visualizations", exist_ok=True)
    plt.savefig(f"{config['experiment']['name']}/visualizations/roc_auc_{save_string}.png")
    return metrics

        
        

        





