import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import slideflow as sf
from sklearn.metrics import auc
from lifelines.utils import concordance_index
from sklearn.calibration import calibration_curve
from sksurv.metrics import cumulative_dynamic_auc
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import seaborn as sns
import logging
import sys

def visualize_activations(config : dict, dataset : str,
                          tfrecord_dir : str, bag_dir : str, target : str, save_string : str):
    """
    Visualize the activations based on the configuration

    Args:
        config: The configuration dictionary
        dataset: The dataset string
        tfrecord_dir: Directory with tfrecords
        bag_dir: Directory with bags
        target: The target variable
        save_string: The save string
    
    Returns:
        None

    """
    #Retrieve features from bags
    dts_ftrs = sf.DatasetFeatures.from_bags(bag_dir)
    #Create slidemap using these features
    slide_map = sf.SlideMap.from_features(dts_ftrs)
    if 'umap' in config['experiment']['visualization'] or 'mosaic' in config['experiment']['visualization']:
        logging.info("Visualizing activations...")
        labels, unique_labels = dataset.labels(target, format='name')
        for index, label in enumerate(unique_labels):
            try:
                #TOFIX: DOES NOT WORK NOW
                slide_map.label_by_preds(index=index)
                slide_map.save_plot(
                    filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_pred_{save_string}_{index}",
                    title=f"Feature UMAP, by prediction of class {index}"
                )

                plt.close() 
            except:
                print(f"Could not visualize UMAP for class {index}")
                pass
            
        slide_map.label_by_slide(labels)
        #Make a new directory inside visualizations
        if not os.path.exists(f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}"):
            os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}")
        slide_map.save_plot(
            filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_umap_label_{save_string}",
            title="Feature UMAP, by label",
            subsample=2000
        )

        plt.close()
    
    if 'mosaic' in config['experiment']['visualization']:

        #TOFIX: DOES NOT WORK NOW!
        logging.info("Building mosaic...")
        # Get list of all directories in the tfrecords dir with full path
        dir_list = [os.path.join(tfrecord_dir, x) for x in os.listdir(tfrecord_dir) if x.endswith('.tfrecord')]
        print(dir_list)
        mosaic = slide_map.build_mosaic(tfrecords=dir_list)
        mosaic.save(
            filename=f"experiments/{config['experiment']['project_name']}/visualizations/slide_level_mosaic_{save_string}.png"
        )
        plt.close()


def plot_roc_auc(y_true, y_pred_prob, save_path, save_string, dataset_type):
    """
    Plots the ROC-AUC curve along with a diagonal baseline and the legend showing the AUC score.

    Args:
        y_true: The true labels
        y_pred_prob: The predicted probabilities
        save_path: Path to save the plot
        save_string: String to append to the saved plot filename
        dataset_type: Type of the dataset (e.g., train, test, validation)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset_type}')
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/roc_auc_{save_string}_{dataset_type}.png")
    plt.close()

    logging.info(f"ROC-AUC plot saved at {save_path}/roc_auc_{save_string}_{dataset_type}.png")

def plot_precision_recall_curve(y_true : np.array, y_pred_prob : np.array, save_path : str, save_string : str, dataset_type : str):
    """
    Plot the precision-recall curve based on the true labels and predicted probabilities

    Args:
        y_true: The true labels
        y_pred_prob: The predicted probabilities
        save_path: The save path
        save_string: The save string
        dataset_type: The dataset type
    
    Returns:
        None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})', color='b')
    #Add baseline precision
    plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], linestyle='--', color='r', label='Baseline (random)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f"{save_path}/precision_recall_curve_{save_string}_{dataset_type}.png")
    plt.close()

def plot_roc_curve_across_splits(rates : list, save_string : str, dataset : str, config : dict):
    """
    Plot the ROC curve across splits based on the rates

    Args:
        rates: The rates
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    fpr_list = [rate[1] for rate in rates]
    tpr_list = [rate[0] for rate in rates]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(len(fpr_list)):
        fpr = fpr_list[i]
        tpr = np.interp(mean_fpr, fpr, tpr_list[i])
        tpr[0] = 0.0
        tprs.append(tpr)
        roc_auc = auc(fpr_list[i], tpr_list[i])
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the ROC curve with shaded standard deviation area
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/roc_auc_{save_string}_{dataset}.png")

def plot_survival_auc_across_folds(results : pd.DataFrame, save_string : str, dataset : str, config : dict):
    """
    Plot the survival ROC-AUC across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None

    """

    fig, ax = plt.subplots()
    all_aucs = []
    times_list = []

    for durations, events, predictions in results:
        # Ensure durations and events are numpy arrays
        durations = np.asarray(durations)
        events = np.asarray(events)
        
        # Create a structured array for survival data
        survival_data = np.array([(e, d) for e, d in zip(events, durations)], dtype=[('event', '?'), ('time', '<f8')])

        # Ensure time points are within the follow-up time range
        max_duration = durations.max()
        times = np.linspace(0, max_duration, 100)
        times_list.append(times)
        
        aucs = []
        for time in times:
            if time >= max_duration:
                continue  # Skip time points that are out of range

            # Calculate AUC and check for validity
            try:
                auc_score = cumulative_dynamic_auc(survival_data, survival_data, predictions, time)[0]
                if not np.isnan(auc_score):
                    aucs.append(auc_score)
            except ZeroDivisionError:
                continue

        if aucs:
            all_aucs.append(np.array(aucs).flatten())  # Ensure the AUC array is flattened
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            ax.plot(times[:len(aucs)], aucs, label=f'AUC (mean = {mean_auc:.2f}, std = {std_auc:.2f})')

    if all_aucs:
        # Determine the maximum length of the AUC lists
        max_len = max(len(auc) for auc in all_aucs)
        
        # Pad the aucs lists to have the same length
        padded_aucs = np.full((len(all_aucs), max_len), np.nan)
        for i, auc in enumerate(all_aucs):
            padded_aucs[i, :len(auc)] = auc

        mean_aucs = np.nanmean(padded_aucs, axis=0)
        std_aucs = np.nanstd(padded_aucs, axis=0)
        times = times_list[0]  # Use times from the first split
        ax.plot(times[:len(mean_aucs)], mean_aucs, label=f'Mean AUC (mean = {np.nanmean(mean_aucs):.2f}, std = {np.nanmean(std_aucs):.2f})', color='blue')

    ax.set_xlabel('Time')
    ax.set_ylabel('AUC')
    ax.set_title('Time-dependent ROC-AUC')
    ax.legend(loc="lower right")

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/survival_roc_auc_{save_string}_{dataset}.png")
    plt.close()


def ensure_monotonic(recall : list, precision : list):
    """
    Ensure that the recall is monotonically increasing and the precision is non-increasing

    Args:
        recall: The recall
        precision: The precision

    Returns:
        The recall and precision, ensuring the conditions are met
    """
    # Ensure that recall is monotonically increasing and precision is non-increasing
    for i in range(1, len(recall)):
        if recall[i] < recall[i - 1]:
            recall[i] = recall[i - 1]
    for i in range(1, len(precision)):
        if precision[i] > precision[i - 1]:
            precision[i] = precision[i - 1]
    return recall, precision

def plot_precision_recall_curve_across_splits(precision_recall_data: list, save_string: str, dataset : str, config: dict):
    """
    Plot the precision-recall curve across splits based on the precision-recall data

    Args:
        precision_recall_data: The precision-recall data, containing tuples of format (precision, recall)
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    # Initialize lists to hold precision and recall values
    precisions = []
    recalls = []

    logging.info(f"Plotting precision-recall curve for {dataset}...")
    logging.info(f"Precision-recall data: {precision_recall_data}")
    # Loop through the data to populate the precision and recall lists
    for precision, recall in precision_recall_data:
        precisions.append(precision)
        recalls.append(recall)
    
    # Convert lists to numpy arrays for easier manipulation
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Compute mean and standard deviation for precision across splits
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)

    # Compute baseline precision
    baseline_precision = np.mean([np.mean(precision) for precision in precisions])

    # Compute AUC for the mean precision-recall curve
    pr_auc = auc(mean_recall, mean_precision)

    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot the mean precision-recall curve
    plt.plot(mean_recall, mean_precision, label=f'Mean Precision-Recall curve (AUC = {pr_auc:.2f})', color='b')
    
    # Fill the area between the mean precision plus/minus the standard deviation
    plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color='b', alpha=0.2, label='Std Deviation')

    # Add baseline precision-recall line
    plt.plot([0, 1], [baseline_precision, baseline_precision], linestyle='--', color='r', label='Baseline (random)')

    # Customize the plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{dataset.name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # Create directory if it doesn't exist
    save_path = f"experiments/{config['experiment']['project_name']}/visualizations"
    os.makedirs(save_path, exist_ok=True)
    
    # Save the plot
    plt.savefig(f"{save_path}/precision_recall_{save_string}_{dataset}.png")
    plt.show()
    plt.close()

def plot_concordance_index_across_folds(results : pd.DataFrame, save_string : str, dataset : str, config : dict):
    """
    Plot the concordance index across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    fig, ax = plt.subplots()
    times = np.linspace(0, max(durations.max() for durations, _, _ in results), 100)
    all_concordances = []

    for durations, events, predictions in results:
        fold_concordances = []
        for time in times:
            mask = durations >= time
            c_index = concordance_index(durations[mask], predictions[mask], events[mask])
            fold_concordances.append(c_index)
        all_concordances.append(fold_concordances)

    mean_concordances = np.mean(all_concordances, axis=0)
    std_concordances = np.std(all_concordances, axis=0)

    ax.plot(times, mean_concordances, marker='o', linewidth=1, label='Mean Concordance Index')
    ax.fill_between(times, mean_concordances - std_concordances, mean_concordances + std_concordances, alpha=0.2)

    ax.set_title('Concordance Index Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concordance Index')
    ax.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/concordance_index_{save_string}_{dataset}.png")
    plt.close()

def plot_predicted_vs_actual_across_folds(results : pd.DataFrame, save_string : str, dataset : str, config : dict):
    """
    Plot the predicted vs actual values across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """
    all_y_true = []
    all_y_pred = []

    for y_true, y_pred in results:
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    mean_y_true = np.mean(all_y_true, axis=0)
    mean_y_pred = np.mean(all_y_pred, axis=0)
    std_y_pred = np.std(all_y_pred, axis=0)

    plt.figure()
    plt.errorbar(mean_y_true, mean_y_pred, yerr=std_y_pred, fmt='o', ecolor='r', capsize=2, label='Mean Predicted vs Actual')
    plt.plot([min(mean_y_true), max(mean_y_true)], [min(mean_y_true), max(mean_y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/predicted_vs_actual_{save_string}_{dataset}.png")
    plt.close()

def plot_calibration(results : pd.DataFrame, save_string : str, dataset : str, config : dict, bins : int = 10):
    """
    Plot the calibration curve based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
        bins: The number of bins

    Returns:
        None
    """
    plt.figure()
    for durations, events, predictions in results:
        # Convert predicted log hazards to predicted risks
        predictions = np.exp(predictions)
        predictions = predictions / (1 + predictions)

        true_prob, pred_prob = calibration_curve(events, predictions, n_bins=bins, strategy='uniform')
        plt.plot(pred_prob, true_prob, marker='o', linewidth=1, label=f'{dataset}')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

def plot_residuals_across_folds(results : pd.DataFrame, save_string : str, dataset : str, config : dict):
    """
    Plot the residuals across folds based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
    
    Returns:
        None
    """

    all_residuals = []

    for y_true, y_pred in results:
        residuals = y_true - y_pred
        all_residuals.append(residuals)

    all_residuals = np.array(all_residuals)
    mean_residuals = np.mean(all_residuals, axis=0)
    std_residuals = np.std(all_residuals, axis=0)

    plt.figure()
    plt.errorbar(np.arange(len(mean_residuals)), mean_residuals, yerr=std_residuals, fmt='o', ecolor='r', capsize=2, label='Mean Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/residuals_{save_string}_{dataset}.png")
    plt.close()

def plot_qq_across_folds(results : pd.DataFrame, save_string : str, dataset : str, config : dict):
    """
    Plot the Q-Q plot across folds based on the results
    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None
    """
    plt.figure()
    for i, (y_true, y_pred) in enumerate(results):
        residuals = y_true - y_pred
        stats.probplot(residuals, dist="norm", plot=plt)
    
    plt.title('Q-Q Plot')
    
    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/qq_{save_string}_{dataset}.png")
    plt.close()

def plot_calibration_across_splits(results : pd.DataFrame, save_string : str, dataset : str, config : dict, bins : int = 10):
    """
    Plot the calibration curve across splits based on the results

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary
        bins: The number of bins

    Returns:
        None
    """
    mean_pred_probs = []
    mean_true_probs = []
    all_pred_probs = []
    all_true_probs = []
    
    for durations, events, predictions in results:
        # Convert predicted log hazards to predicted risks
        predictions = np.exp(predictions)
        predictions = predictions / (1 + predictions)

        true_prob, pred_prob = calibration_curve(events, predictions, n_bins=bins, strategy='uniform')
        mean_pred_probs.append(pred_prob)
        mean_true_probs.append(true_prob)
        all_pred_probs.append(pred_prob)
        all_true_probs.append(true_prob)

    # Calculate the mean and standard error across splits
    mean_pred_probs = np.mean(mean_pred_probs, axis=0)
    mean_true_probs = np.mean(mean_true_probs, axis=0)
    std_true_probs = np.std(all_true_probs, axis=0)
    
    plt.figure()
    plt.plot(mean_pred_probs, mean_true_probs, marker='o', linewidth=1, label='Mean Calibration')
    plt.fill_between(mean_pred_probs, mean_true_probs - std_true_probs, mean_true_probs + std_true_probs, alpha=0.2, label='±1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

#TODO: Implement function that outputs the top 5 most- and least attented tiles
"""
def get_top5_annotation_tiles_per_class(attention_map, bag_index, tile_directory, slide_name, diagnosis):
    #Get the top 5 attention values
    top5_attention_values = np.argsort(attention_map)[::-1][:5]
    print(f"Top 5 attention indices: {top5_attention_values}")
    top5_least_attended = np.argsort(attention_map)[:5]

    #Get the corresponding tile coordinates from bag_index
    top5_tile_coordinates = bag_index[top5_attention_values]
    print(f"Top 5 tile coordinates: {top5_tile_coordinates}")
    top5_least_attended_coordinates = bag_index[top5_least_attended]

    #Get the corresponding tile paths
    top5_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_tile_coordinates]
    top5_least_attended_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_least_attended_coordinates]
    print(f"Top 5 tile paths: {top5_tiles}")

    #Construct top 5 tiles directory
    os.makedirs(f"top5_tiles", exist_ok=True)

    #Plot the top 5 tiles using matplotlib
    import matplotlib.pyplot as plt
    import cv2
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, tile in enumerate(top5_tiles):
        img = Image.open(tile)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        axs[i].imshow(img)
        axs[i].axis('off')
    #Set global title above the subplots
    plt.suptitle(f"Top 5 most attended tiles ({diagnosis} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"top5_tiles/{diagnosis}_{slide_name}_top5_tiles.png")
    plt.close()
"""