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
import warnings


def get_continuous_preds(predictions):
    """
    Convert discrete survival predictions (N, n_bins) to a single continuous
    time or risk estimate by treating bin indices as 1..n_bins and computing
    an expected value. If predictions is already shape (N,), it is returned unchanged.

    If your model outputs probabilities across n_bins for each patient,
    each row i might sum to ~1. If not, we row-wise normalize. Then:
       cont_pred[i] = sum_{k=1..n_bins}( prob_{i,k} * k ).
    """
    if predictions.ndim == 1:
        # Already continuous
        return predictions
    
    elif predictions.ndim == 2:
        n_bins = predictions.shape[1]
        # Our bin indices = 1..n_bins
        bin_indices = np.arange(1, n_bins + 1, dtype=float)

        # Normalize row-wise (in case they don't already sum to 1)
        row_sums = predictions.sum(axis=1, keepdims=True)
        probs = predictions / np.clip(row_sums, 1e-12, None)

        # Weighted average
        return np.sum(probs * bin_indices, axis=1)
    
    else:
        raise ValueError(f"Invalid predictions shape {predictions.shape}. "
                         "Must be (N,) or (N, n_bins).")



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
    
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
from sklearn.metrics import roc_auc_score

def plot_survival_auc_across_folds(results_per_split, save_string, dataset, config):
    """
    Plot time-dependent ROC AUC curves for continuous survival predictions (log hazard)
    across folds using cumulative dynamic AUC.
    
    This function assumes that the predictions are continuous log-hazard values.
    
    Args:
        results_per_split (list):
            List of tuples (durations, events, raw_preds) for each fold.
              - durations: array-like of survival times.
              - events: array-like of event indicators (0/1 or bool).
              - raw_preds: 1D array of continuous log-hazard predictions.
        save_string (str):
            Identifier string for saving the plot.
        dataset (str):
            Name of the dataset.
        config (dict):
            Configuration dictionary with at least config['experiment']['project_name'].
    
    Returns:
        None. The plot is saved to a file.
    """

    if config['experiment']['task'] != 'survival':
        logging.warning("Skipping survival ROC AUC plot: Task is not 'survival'.")
        return
    # Collect all durations from all folds to build a common time grid.
    all_durations = []
    for durations, _, _ in results_per_split:
        durations = np.asarray(durations).flatten().astype(float)
        all_durations.append(durations)
    all_durations = np.concatenate(all_durations)
    global_min = all_durations.min()
    global_max = all_durations.max()
    
    # Create a uniform time grid spanning the global minimum and maximum times.
    time_grid = np.linspace(global_min, global_max, num=100)
    
    fold_aucs = []
    # Process each fold.
    for fold_idx, (durations, events, raw_preds) in enumerate(results_per_split):
        durations = np.asarray(durations).flatten().astype(float)
        events = np.asarray(events).flatten().astype(bool)
        raw_preds = np.asarray(raw_preds).flatten()
        
        if raw_preds.ndim != 1:
            raise ValueError(f"Fold {fold_idx}: Expected continuous 1D predictions (log hazard).")
        
        # Build a structured array for survival data.
        dtype = np.dtype([('event', bool), ('time', float)])
        survival_data = np.array([(e, t) for e, t in zip(events, durations)], dtype=dtype)
        
        # For each time in our grid, compute AUC only for time points strictly between the fold's min and max.
        t_min = durations.min()
        t_max = durations.max()
        aucs = np.full_like(time_grid, np.nan, dtype=float)
        valid_mask = (time_grid > t_min) & (time_grid < t_max)
        if valid_mask.any():
            try:
                # cumulative_dynamic_auc computes AUC scores at the requested times.
                auc_scores, _ = cumulative_dynamic_auc(
                    survival_data,  # training data (here using the same set)
                    survival_data,  # test data
                    raw_preds,      # risk scores (log hazard predictions)
                    time_grid[valid_mask]
                )
                aucs[valid_mask] = auc_scores
            except Exception as e:
                logging.warning(f"Fold {fold_idx}: Error computing cumulative dynamic AUC: {e}")
        fold_aucs.append(aucs)
    
    fold_aucs = np.array(fold_aucs)  # shape: (n_folds, len(time_grid))
    mean_auc = np.nanmean(fold_aucs, axis=0)
    std_auc = np.nanstd(fold_aucs, axis=0)
    
    # Plot the mean AUC over time with ±1 standard deviation error bands.
    plt.figure(figsize=(8, 6))
    plt.plot(time_grid, mean_auc, color='blue', lw=2, label='Mean AUC')
    plt.fill_between(time_grid, mean_auc - std_auc, mean_auc + std_auc, color='blue', alpha=0.2,
                     label='±1 std. dev.')
    plt.axhline(0.5, color='red', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('Time')
    plt.ylabel('ROC AUC')
    plt.title('Time-dependent ROC AUC (Continuous Predictions)')
    plt.legend(loc="lower right")
    
    # Save the plot.
    project_name = config['experiment']['project_name']
    save_dir = os.path.join("experiments", project_name, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"roc_auc_{save_string}_{dataset}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved survival ROC AUC plot to {save_path}")


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


def plot_concordance_index_across_folds(
    results_per_split: list,  # list of (durations, events, predictions)
    save_string: str,
    dataset: str,
    config: dict
):
    """
    Plot c-index over time (approx) across folds.
    Procedure:
      - Convert predictions to continuous if needed (discrete -> expected bin index).
      - Collect all durations across folds -> unique time points
      - For each fold and each time t, mask patients with durations >= t
      - Compute c-index among those patients
      - Average across folds
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from lifelines.utils import concordance_index

    # Collect all durations
    all_durations = []
    for (durations, events, _) in results_per_split:
        all_durations.append(durations)
    unique_durations = np.sort(np.unique(np.concatenate(all_durations).astype(float)))

    cindexes_per_fold = []

    for fold_idx, (durations, events, raw_preds) in enumerate(results_per_split):
        durations = durations.reshape(-1).astype(float)
        events    = events.astype(bool)

        # Discrete => continuous
        cont_preds = get_continuous_preds(raw_preds)

        cindexes = []
        for t in unique_durations:
            # At-risk: durations >= t
            mask = (durations >= t)
            if mask.sum() < 2:
                cindexes.append(np.nan)
                continue
            try:
                cind = concordance_index(
                    durations[mask],
                    cont_preds[mask],
                    events[mask]
                )
                cindexes.append(cind)
            except ZeroDivisionError:
                cindexes.append(np.nan)

        cindexes_per_fold.append(cindexes)

    cindexes_per_fold = np.array(cindexes_per_fold, dtype=float)  # shape: (n_folds, len(unique_durations))
    mean_cindex = np.nanmean(cindexes_per_fold, axis=0)
    std_cindex  = np.nanstd(cindexes_per_fold,  axis=0)

    plt.figure()
    plt.plot(unique_durations, mean_cindex, marker='o', linewidth=1, markersize=3, label='Mean C-Index')
    plt.fill_between(unique_durations,
                     mean_cindex - std_cindex,
                     mean_cindex + std_cindex,
                     alpha=0.2, label='±1 std. dev.')

    plt.xlim([unique_durations[0], unique_durations[-1]])
    plt.ylim([0, 1])
    plt.title('Concordance Index Over Time (Pooled Across Folds)')
    if config['experiment']['task'] == 'survival':
        plt.xlabel('Time')
    elif config['experiment']['task'] == 'survival_discrte':
        plt.xlabel('Time bin')
    plt.ylabel('C-Index')
    plt.legend()

    viz_dir = f"experiments/{config['experiment']['project_name']}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    outpath = os.path.join(viz_dir, f"concordance_index_{save_string}_{dataset}.png")
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Concordance index plot saved to {outpath}")

    
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


def plot_calibration_curve_across_splits(results_per_split, save_string, dataset, config, n_bins=10):
    """
    Plot a calibration curve for continuous survival predictions (log hazard) across splits.
    
    The continuous log-hazard predictions are converted to predicted probabilities via:
         p = exp(log_h) / (1 + exp(log_h))
    Calibration is then computed by comparing these probabilities with the observed binary event outcomes,
    ignoring censoring.
    
    Args:
        results_per_split (list):
            List of tuples (durations, events, raw_preds) for each fold.
              - durations: array-like of survival times.
              - events: array-like of event indicators (0/1).
              - raw_preds: 1D array of continuous log-hazard predictions. In some cases, these may be
                           provided as a 2D array with a singleton dimension.
        save_string (str):
            Identifier string for saving the plot.
        dataset (str):
            Name of the dataset.
        config (dict):
            Configuration dictionary with at least config['experiment']['project_name'].
        n_bins (int):
            Number of bins to use for the calibration curve.
    
    Returns:
        None. The calibration plot is saved to a file.
    """

    if config['experiment']['task'] != 'survival':
        logging.warning("Skipping calibration curve plot: Task is not 'survival'.")
        return
        
    # Define a common grid on [0, 1] for interpolation.
    x_common = np.linspace(0, 1, 50)
    interp_curves = []
    
    for fold_idx, (_, events, raw_preds) in enumerate(results_per_split):
        # Process events as before.
        events = np.asarray(events).flatten().astype(int)
        
        # Convert raw_preds to a numpy array.
        raw_preds = np.asarray(raw_preds)
        # If raw_preds is 2D, squeeze it appropriately.
        if raw_preds.ndim == 2:
            if raw_preds.shape[0] == 1:
                # If shape is (1, n_samples), squeeze the first axis.
                raw_preds = raw_preds.squeeze(axis=0)
            elif raw_preds.shape[1] == 1:
                # If shape is (n_samples, 1), squeeze the second axis.
                raw_preds = raw_preds.squeeze(axis=1)
            else:
                # If there are multiple columns, warn and take the first column.
                logging.warning(f"raw_preds for fold {fold_idx} has shape {raw_preds.shape}. Using the first column.")
                raw_preds = raw_preds[:, 0]
        else:
            # Ensure a 1D array.
            raw_preds = raw_preds.flatten()
        
        # Convert log-hazard predictions to probabilities.
        # (This is equivalent to the logistic function: 1/(1 + exp(-log_h)).)
        p = np.exp(raw_preds) / (1.0 + np.exp(raw_preds))
        
        # Compute the calibration curve for this fold.
        # calibration_curve returns (fraction_of_positives, mean_predicted_value)
        frac_pos, mean_pred = calibration_curve(events, p, n_bins=n_bins, strategy='uniform')
        
        # Ensure the calibration points are in increasing order.
        sort_idx = np.argsort(mean_pred)
        mean_pred = mean_pred[sort_idx]
        frac_pos = frac_pos[sort_idx]
        
        # Interpolate the calibration curve onto the common x-axis.
        if len(mean_pred) < 2:
            # If only one bin is available, fill with the constant value.
            interp_vals = np.full_like(x_common, frac_pos[0] if len(frac_pos) > 0 else np.nan, dtype=float)
        else:
            interp_vals = np.interp(x_common, mean_pred, frac_pos, left=np.nan, right=np.nan)
        interp_curves.append(interp_vals)
    
    interp_curves = np.array(interp_curves)
    mean_calib = np.nanmean(interp_curves, axis=0)
    std_calib = np.nanstd(interp_curves, axis=0)
    
    # Plot the mean calibration curve with error bands.
    plt.figure(figsize=(8, 6))
    plt.plot(x_common, mean_calib, marker='o', label='Mean Calibration', color='blue')
    plt.fill_between(x_common, mean_calib - std_calib, mean_calib + std_calib,
                     color='blue', alpha=0.2, label='±1 std')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve (Continuous Predictions)')
    plt.legend(loc='upper left')
    
    # Save the plot.
    project_name = config['experiment']['project_name']
    save_dir = os.path.join("experiments", project_name, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"calibration_{save_string}_{dataset}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved calibration curve plot to {save_path}")

    
def plot_kaplan_meier_curves_across_folds(results_per_split, save_string, dataset, config):
    """
    Plot Kaplan–Meier curves for high- vs low-risk groups, pooling data across folds.

    For each fold:
      1. Convert raw predictions to continuous risk scores.
      2. Compute the per-fold median risk and split patients into high- vs low-risk groups.
    The data from all folds are then pooled to fit Kaplan–Meier curves for each risk group.
    A log-rank test is performed and its p-value is annotated on the plot.

    Args:
        results_per_split (list): List of tuples (durations, events, raw_preds) for each fold.
        save_string (str): String used to name the output plot.
        dataset (str): Dataset identifier.
        config (dict): Configuration dictionary (expects keys such as
            config['experiment']['project_name'] and config['experiment']['task']).
    """
    # Define directory and file path for saving the plot.
    viz_dir = os.path.join("experiments", config['experiment']['project_name'], "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    outpath = os.path.join(viz_dir, f"kaplan_meier_{save_string}_{dataset}.png")

    # Lists to hold durations and events for the two risk groups.
    all_durations_high, all_events_high = [], []
    all_durations_low,  all_events_low  = [], []

    # Process each fold’s results.
    for (durations, events, raw_preds) in results_per_split:
        durations = np.asarray(durations).flatten().astype(float)
        events = np.asarray(events).flatten().astype(bool)

        # Convert raw predictions to continuous risk scores.
        cont_preds = get_continuous_preds(raw_preds)

        # Determine the per-fold median and split patients.
        median_pred = np.median(cont_preds)
        high_mask = cont_preds >= median_pred
        low_mask = cont_preds < median_pred

        all_durations_high.append(durations[high_mask])
        all_events_high.append(events[high_mask])
        all_durations_low.append(durations[low_mask])
        all_events_low.append(events[low_mask])

    # Pool the data across folds.
    pooled_dur_high = np.concatenate(all_durations_high)
    pooled_evt_high = np.concatenate(all_events_high)
    pooled_dur_low = np.concatenate(all_durations_low)
    pooled_evt_low = np.concatenate(all_events_low)

    # Fit Kaplan–Meier models.
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    # Perform log-rank test between high- and low-risk groups.
    results_logrank = logrank_test(
        pooled_dur_high, pooled_dur_low,
        event_observed_A=pooled_evt_high,
        event_observed_B=pooled_evt_low
    )
    p_value = results_logrank.p_value

    plt.figure(figsize=(8, 6))
    kmf_high.fit(pooled_dur_high, pooled_evt_high, label='High Risk (Pooled)')
    kmf_high.plot(ci_show=True, linestyle='--')
    kmf_low.fit(pooled_dur_low, pooled_evt_low, label='Low Risk (Pooled)')
    kmf_low.plot(ci_show=True, linestyle='-')

    plt.title(f"Kaplan–Meier Curves (High vs Low Risk) - {dataset}")
    # Choose xlabel based on the task type.
    xlabel = 'Time' if config['experiment'].get('task', 'survival') == 'survival' else 'Time bin'
    plt.xlabel(xlabel)
    plt.ylabel('Survival Probability')
    plt.legend()

    # Annotate the log-rank p-value.
    p_text = f"p = {p_value:.3g}" if p_value >= 0.001 else "p < 0.001"
    plt.text(0.95, 0.95, p_text,
             transform=plt.gca().transAxes,
             ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Kaplan–Meier curves saved to {outpath}")
    
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
        draw.text((10, 10), f"Score: {attention[top5_attention_indices[i]]:.4f}", fill="black", font=font)

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