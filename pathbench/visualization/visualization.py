import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import slideflow as sf
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import auc
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.calibration import calibration_curve
from sksurv.metrics import cumulative_dynamic_auc
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import seaborn as sns
import logging
import sys
from matplotlib import cm
from matplotlib.patches import Patch


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
    plt.close()
    
def plot_survival_auc_across_folds(results: pd.DataFrame, save_string: str, dataset: str, config: dict):
    """
    Plot the overall survival ROC-AUC across folds with standard deviation as error margins.

    Args:
        results: The results
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None
    """
    import warnings
    all_aucs = []

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)

    # Collect all durations to determine the overall max_duration
    all_durations = []
    for durations, events, predictions in results:
        all_durations.extend(durations)
    all_durations = np.asarray(all_durations)
    max_duration = all_durations.max()

    # Define common time points
    times = np.linspace(0, max_duration, 100)

    for fold_idx, (durations, events, predictions) in enumerate(results):
        print(f"Processing fold {fold_idx}")
        # Ensure durations and events are numpy arrays
        durations = np.asarray(durations)
        events = np.asarray(events)
        predictions = np.asarray(predictions)

        # Check for NaNs and constants in predictions
        if np.isnan(predictions).any():
            print(f"Fold {fold_idx}: Predictions contain NaN values.")
            continue
        if np.all(predictions == predictions[0]):
            print(f"Fold {fold_idx}: Predictions are constant.")
            continue

        # Create a structured array for survival data
        survival_data = np.array(
            list(zip(events.astype(bool), durations)),
            dtype=[('event', bool), ('time', float)]
        )

        aucs = []

        for time_idx, time in enumerate(times):
            # Skip time points beyond maximum duration in the fold
            if time >= durations.max():
                aucs.append(np.nan)
                continue

            # Count events up to current time
            event_count = np.sum((durations <= time) & (events == 1))
            if event_count == 0:
                print(f"Fold {fold_idx}, Time {time}: No events occurred. AUC cannot be computed.")
                aucs.append(np.nan)
                continue

            # Calculate AUC
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    auc_scores, _ = cumulative_dynamic_auc(
                        survival_train=survival_data,
                        survival_test=survival_data,
                        estimate=predictions,
                        times=[time]
                    )
                    auc_score = auc_scores[0]
                    if not np.isnan(auc_score):
                        aucs.append(auc_score)
                    else:
                        print(f"Fold {fold_idx}, Time {time}: AUC is NaN.")
                        aucs.append(np.nan)
            except Exception as e:
                print(f"Fold {fold_idx}, Time {time}: Exception occurred: {e}")
                aucs.append(np.nan)
                continue

        all_aucs.append(aucs)

    # Now compute mean and std AUCs at each time point
    all_aucs = np.array(all_aucs)  # Shape: (n_folds, n_times)

    # Identify time points where not all values are NaN
    valid_time_points = ~np.all(np.isnan(all_aucs), axis=0)
    if not np.any(valid_time_points):
        print("All time points have NaN AUCs. Cannot compute mean AUC.")
        return

    # Get valid times
    valid_times = times[valid_time_points]

    # Compute mean and std only for valid time points
    mean_aucs = np.nanmean(all_aucs[:, valid_time_points], axis=0)
    std_aucs = np.nanstd(all_aucs[:, valid_time_points], axis=0)

    fig, ax = plt.subplots()

    ax.plot(valid_times, mean_aucs, label='Mean AUC', color='blue')
    ax.fill_between(valid_times, mean_aucs - std_aucs, mean_aucs + std_aucs, color='blue', alpha=0.2)

    # Add random model baseline (0.5)
    ax.plot(valid_times, [0.5] * len(valid_times), 'r--', label='Random Model (AUC = 0.5)')

    ax.set_xlabel('Time')
    ax.set_ylabel('AUC')
    ax.set_title('Time-dependent ROC-AUC (Overall)')
    ax.legend(loc="lower right")

    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/survival_roc_auc_{save_string}_{dataset}_overall.png")
    plt.close()


def plot_precision_recall_across_splits(pr_per_split: list, save_string: str, dataset: str, config: dict):
    """
    Plot the precision-recall curve across splits based on the results

    Args:
        pr_per_split: List of precision-recall curve results per split
        save_string: The save string
        dataset: The dataset
        config: The configuration dictionary

    Returns:
        None
    """
    fig, ax = plt.subplots()
    mean_precisions = []
    mean_recalls = []
    all_precisions = []
    all_recalls = []

    for split_idx, pr_tuple in enumerate(pr_per_split):
        # Check and unpack the tuple
        if isinstance(pr_tuple, list) and len(pr_tuple) == 1:
            pr_tuple = pr_tuple[0]

        if not isinstance(pr_tuple, tuple) or len(pr_tuple) != 2:
            raise ValueError(f"Expected a tuple of (precision, recall) for split {split_idx + 1}, but got {pr_tuple}")

        precisions, recalls = pr_tuple
        mean_precisions.append(np.interp(np.linspace(0, 1, 100), recalls[::-1], precisions[::-1]))
        mean_recalls.append(np.linspace(0, 1, 100))
        all_precisions.append(precisions)
        all_recalls.append(recalls)

    mean_precision = np.mean(mean_precisions, axis=0)
    std_precision = np.std(mean_precisions, axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.plot(mean_recalls[0], mean_precision, label='Mean Precision-Recall', color='blue')
    ax.fill_between(mean_recalls[0], mean_precision - std_precision, mean_precision + std_precision, color='blue', alpha=0.2, label='Std Dev')

    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="lower left")

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/precision_recall_{save_string}_{dataset}.png")
    plt.close()

def plot_concordance_index_across_folds(results: pd.DataFrame, save_string: str, dataset: str, config: dict):
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
            if mask.sum() == 0:
                fold_concordances.append(np.nan)
            else:
                try:
                    c_index = concordance_index(durations[mask], predictions[mask], events[mask])
                    fold_concordances.append(c_index)
                except ZeroDivisionError:
                    fold_concordances.append(np.nan)
        all_concordances.append(fold_concordances)

    all_concordances = np.array(all_concordances)
    mean_concordances = np.nanmean(all_concordances, axis=0)
    std_concordances = np.nanstd(all_concordances, axis=0)

    
    ax.plot(times, mean_concordances, marker='o', linewidth=1, label='Mean Concordance Index', markersize=3)
    ax.fill_between(times, mean_concordances - std_concordances, mean_concordances + std_concordances, alpha=0.2)
    ax.set_xlim([0, times[-1]])
    ax.set_ylim([0, 1])
    ax.set_title('Concordance Index Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concordance Index')
    ax.legend()



    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/concordance_index_{save_string}_{dataset}.png")
    plt.close()

def plot_predicted_vs_actual_across_folds(results: pd.DataFrame, save_string: str, dataset: str, config: dict):
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
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    # Determine the maximum length of the y_true and y_pred
    max_len = max(max(len(y) for y in all_y_true), max(len(y) for y in all_y_pred))

    # Pad the y_true and y_pred to the maximum length with np.nan
    padded_y_true = np.full((len(all_y_true), max_len), np.nan)
    padded_y_pred = np.full((len(all_y_pred), max_len), np.nan)

    for i, (y_true, y_pred) in enumerate(zip(all_y_true, all_y_pred)):
        padded_y_true[i, :len(y_true)] = y_true
        padded_y_pred[i, :len(y_pred)] = y_pred

    mean_y_true = np.nanmean(padded_y_true, axis=0)
    mean_y_pred = np.nanmean(padded_y_pred, axis=0)
    std_y_pred = np.nanstd(padded_y_pred, axis=0)

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
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

def plot_residuals_across_folds(results: pd.DataFrame, save_string: str, dataset: str, config: dict):
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
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred
        all_residuals.append(residuals)

    # Determine the maximum length of the residuals
    max_len = max(len(residual) for residual in all_residuals)

    # Pad the residuals to the maximum length with np.nan
    padded_residuals = np.full((len(all_residuals), max_len), np.nan)
    for i, residual in enumerate(all_residuals):
        padded_residuals[i, :len(residual)] = residual

    mean_residuals = np.nanmean(padded_residuals, axis=0)
    std_residuals = np.nanstd(padded_residuals, axis=0)

    plt.figure()
    plt.errorbar(np.arange(len(mean_residuals)), mean_residuals, yerr=std_residuals, fmt='o', ecolor='r', capsize=2, label='Mean Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Index')
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

def plot_calibration_across_splits(results: pd.DataFrame, save_string: str, dataset: str, config: dict, bins: int = 10):
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

        # Interpolate to a common set of bins
        common_bins = np.linspace(0, 1, bins)
        true_prob_interp = np.interp(common_bins, pred_prob, true_prob)
        pred_prob_interp = common_bins

        mean_pred_probs.append(pred_prob_interp)
        mean_true_probs.append(true_prob_interp)
        all_pred_probs.append(pred_prob_interp)
        all_true_probs.append(true_prob_interp)

    # Calculate the mean and standard deviation across splits
    mean_pred_probs = np.mean(mean_pred_probs, axis=0)
    mean_true_probs = np.mean(mean_true_probs, axis=0)
    std_true_probs = np.std(all_true_probs, axis=0)
    
    plt.figure()
    plt.plot(mean_pred_probs, mean_true_probs, marker='o', linewidth=1, label='Mean Calibration')
    plt.fill_between(mean_pred_probs, mean_true_probs - std_true_probs, mean_true_probs + std_true_probs, alpha=0.2, label='±1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.legend()

    os.makedirs(f"experiments/{config['experiment']['project_name']}/visualizations", exist_ok=True)
    plt.savefig(f"experiments/{config['experiment']['project_name']}/visualizations/calibration_{save_string}_{dataset}.png")
    plt.close()

def plot_kaplan_meier_curves_across_folds(results: pd.DataFrame, save_string: str, dataset: str, config: dict):
    """
    Plot the Kaplan-Meier curves by averaging the results across folds, splitting patients into high-risk and low-risk groups,
    and include the p-value from the log-rank test.

    Args:
        results: List of tuples containing (durations, events, predictions) for each fold.
        save_string: String used to save the plot.
        dataset: Dataset name (e.g., 'train', 'test', 'val').
        config: The configuration dictionary.

    Returns:
        None
    """
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    # Initialize lists to collect pooled durations and events for high-risk and low-risk groups
    all_durations_high, all_events_high = [], []
    all_durations_low, all_events_low = [], []

    # Iterate over each fold and collect high-risk and low-risk groups
    for fold_idx, (durations, events, predictions) in enumerate(results):
        # Split into high-risk and low-risk groups using the median of the predictions
        median_prediction = np.median(predictions)

        high_risk_mask = predictions >= median_prediction
        low_risk_mask = predictions < median_prediction

        # Append the durations and events for high-risk group
        all_durations_high.append(durations[high_risk_mask])
        all_events_high.append(events[high_risk_mask])

        # Append the durations and events for low-risk group
        all_durations_low.append(durations[low_risk_mask])
        all_events_low.append(events[low_risk_mask])

    # Concatenate the pooled durations and events across all folds
    pooled_durations_high = np.concatenate(all_durations_high)
    pooled_events_high = np.concatenate(all_events_high)

    pooled_durations_low = np.concatenate(all_durations_low)
    pooled_events_low = np.concatenate(all_events_low)

    # Perform the log-rank test to compute the p-value
    results_logrank = logrank_test(
        pooled_durations_high,
        pooled_durations_low,
        event_observed_A=pooled_events_high,
        event_observed_B=pooled_events_low
    )
    p_value = results_logrank.p_value

    # Create a plot for the Kaplan-Meier curves averaged over folds
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fit and plot the Kaplan-Meier curve for the high-risk group
    kmf_high.fit(
        pooled_durations_high,
        event_observed=pooled_events_high,
        label='High Risk (Averaged)'
    )
    kmf_high.plot(
        ax=ax,
        ci_show=True,
        show_censors=True,
        linestyle='--'
    )

    # Fit and plot the Kaplan-Meier curve for the low-risk group
    kmf_low.fit(
        pooled_durations_low,
        event_observed=pooled_events_low,
        label='Low Risk (Averaged)'
    )
    kmf_low.plot(
        ax=ax,
        ci_show=True,
        show_censors=True,
        linestyle='-'
    )

    # Customize the plot
    ax.set_title(
        'Kaplan-Meier Curves for High-Risk and Low-Risk Groups (Averaged Across Folds)'
    )
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Survival Probability')
    ax.legend(title="Risk Groups", loc='best')

    # Format the p-value for display
    if p_value < 0.001:
        p_value_text = 'p < 0.001'
    else:
        p_value_text = f'p = {p_value:.3f}'

    # Add the p-value text to the plot
    ax.text(
        0.95,
        0.95,
        p_value_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5)
    )

    # Ensure the visualization directory exists
    os.makedirs(
        f"experiments/{config['experiment']['project_name']}/visualizations",
        exist_ok=True
    )

    # Save the Kaplan-Meier plot
    plot_path = (
        f"experiments/{config['experiment']['project_name']}/visualizations/"
        f"kaplan_meier_averaged_{save_string}_{dataset}.png"
    )
    plt.savefig(plot_path)
    plt.close()

    logging.info(
        f"Averaged Kaplan-Meier curves for high-risk and low-risk groups saved to {plot_path}"
    )

def plot_top5_attended_tiles_per_class(slide_name : str, attention_file : str, tfr_directory : str, output_dir : str,
                                       annotation_df : pd.DataFrame, target : str, dataset : str, split : str, save_string : str):
    """
    Plot the top 5 most attended tiles and the top 5 least attended tiles for a given slide

    Args:
        slide_name: The slide name to get the top 5 attended tiles for
        attention_file: The attention file containing the attention map
        tfr_directory: The TFRecord directory
        output_dir: The output directory
        annotation_df: The annotation DataFrame
        target: The target variable
        dataset: The dataset string (val,test,train)
        split: The current cv-split
        save_string: The save string
    
    Returns:
        None
    """
    # Find the corresponding .npz and .tfrecord files
    index_path = os.path.join(tfr_directory, f"{slide_name}.index.npz")
    tfr_path = os.path.join(tfr_directory, f"{slide_name}.tfrecords")

    # Validate the existence of the files
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not os.path.exists(tfr_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfr_path}")

    # Retrieve the label for the slide from the annotation DataFrame
    label_row = annotation_df.loc[annotation_df['slide'] == slide_name]
    if label_row.empty:
        raise ValueError(f"No label found for slide: {slide_name} in the annotation DataFrame")
    
    label = label_row[target].values[0]

    # Load the attention map
    attention = attention_file['arr_0']

    # Load the TFRecord and its index
    tfr = sf.TFRecord(tfr_path)
    
    # Load the index file, which contains the tile coordinates
    bag_index = np.load(index_path)
    tile_coordinates = bag_index['locations']  # This contains the (x, y) coordinates of the tiles

    # Get the top 5 attention values and their indices
    top5_attention_indices = np.argsort(attention)[::-1][:5]
    top5_least_attended_indices = np.argsort(attention)[:5]

    # Get the corresponding tile coordinates from the index file
    top5_tile_coordinates = tile_coordinates[top5_attention_indices]
    top5_least_attended_coordinates = tile_coordinates[top5_least_attended_indices]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot the top 5 most attended tiles with attention scores
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, coord in enumerate(top5_tile_coordinates):
        slide, img = tfr.get_record_by_xy(coord[0], coord[1], decode=True)

        # Convert from Tensor to NumPy array if necessary
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  # No permutation needed, assuming the shape is already [256, 256, 3]

        # Convert to image (RGB assumed)
        img = Image.fromarray(img.astype(np.uint8))  # Convert directly to image

        # Add attention score text to image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Score: {attention[top5_attention_indices[i]]:.4f}", fill="red", font=font)

        # Plot the image
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.suptitle(f"Top 5 most attended tiles ({dataset} {label} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_{split}_{label}_{slide_name}_{save_string}_top5_tiles.png")
    plt.close()

    # Plot the top 5 least attended tiles with attention scores
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, coord in enumerate(top5_least_attended_coordinates):
        slide, img = tfr.get_record_by_xy(coord[0], coord[1], decode=True)


        # Convert from Tensor to NumPy array if necessary
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  # No permutation needed, assuming the shape is already [256, 256, 3]

        # Convert to image (RGB assumed)
        img = Image.fromarray(img.astype(np.uint8))  # Convert directly to image

        # Add attention score text to image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Score: {attention[top5_least_attended_indices[i]]:.4f}", fill="red", font=font)

        # Plot the image
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.suptitle(f"Top 5 least attended tiles ({dataset} {label} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_{split}_{label}_{slide_name}_{save_string}_least_attended_tiles.png")
    plt.close()

def visualize_benchmarking_results_for_all_metrics(df: pd.DataFrame, config: dict, dataset: str):
    import os
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    
    # Dynamically select columns that are not float-valued for the combination
    non_float_columns = df.select_dtypes(exclude=['float']).columns.tolist()
    
    # Combine the non-float columns to form the combination string
    df['combination'] = df[non_float_columns].astype(str).agg(' | '.join, axis=1)
    
    # Identify all the metrics (columns that end with '_mean')
    metric_columns = [col for col in df.columns if col.endswith('_mean')]
    
    # Create the directory to save plots
    results_dir = f"experiments/{config['experiment']['project_name']}/results/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Loop through each metric and create a separate plot
    for metric in metric_columns:
        # Get the corresponding standard deviation column
        metric_std = metric.replace('_mean', '_std')
        
        # Sort the dataframe by the current metric
        df_sorted = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
        
        # Create a figure for each metric
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get unique combinations and assign colors
        unique_combinations = df_sorted['combination'].unique()
        num_combinations = len(unique_combinations)
        
        # Generate a colormap with enough colors
        if num_combinations <= 20:
            cmap = cm.get_cmap('tab20', num_combinations)
        else:
            cmap = cm.get_cmap('hsv', num_combinations)
        
        colors = cmap(np.linspace(0, 1, num_combinations))
        
        # Map combinations to colors
        color_map = {combination: colors[i] for i, combination in enumerate(unique_combinations)}
        
        # Map colors to combinations for the bars
        bar_colors = df_sorted['combination'].map(color_map)
        
        # Plot the bar plot with numeric y values (use range(len(df_sorted)) for y-axis)
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], xerr=df_sorted[metric_std], 
                       color=bar_colors, edgecolor='black', capsize=5, 
                       error_kw={'elinewidth': 1.5, 'capthick': 2})
        
        # Set y-ticks to correspond to the number of bars
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['combination'], fontsize=8)  # Smaller text for y-tick labels
        
        # Invert y-axis so the highest value is on top
        ax.invert_yaxis()

        # Add labels and title
        ax.set_xlabel(f'{metric.replace("_mean", "").capitalize()} (Mean)')
        ax.set_title(f'Benchmarking Results: {metric.replace("_mean", "").capitalize()}')
        
        # Create legend
        # To avoid duplicates in legend, we can create a mapping from combination to color once
        legend_handles = [Patch(facecolor=color_map[combination], edgecolor='black', label=combination) 
                          for combination in unique_combinations]
        ax.legend(handles=legend_handles, loc='best', fontsize=8)
        
        # Adjust layout to make sure labels fit
        plt.tight_layout()
        
        # Save each plot to the results directory
        metric_name = metric.replace('_mean', '')
        plot_path = os.path.join(results_dir, f"{metric_name}_{dataset}_benchmarking_results.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()  # Close the figure after saving to free memory

    print(f"All plots saved to {results_dir}")