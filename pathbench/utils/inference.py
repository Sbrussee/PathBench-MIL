#!/usr/bin/env python
import argparse
import os
import sys
import logging
import json
import yaml
import numpy as np
from slideflow import Dataset
from slideflow.mil import predict_slide
from slideflow.util import location_heatmap
from pathbench.models import feature_extractors, slide_level_predictors

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a MIL model and optionally generate attention heatmaps."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved MIL model directory (must contain mil_params.json, models/, etc.)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the PathBench configuration file (e.g., config.yaml)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--slide",
        type=str,
        help="Path to a single slide file (e.g., .svs)"
    )
    group.add_argument(
        "--slide_dir",
        type=str,
        help="Path to a directory containing slide files"
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="If set, also saves attention heatmap images."
    )

    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="If set, also saves uncertainty values."
    )

    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        help="Interpolation strategy for smoothing heatmap."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap for heatmap."
    )
    parser.add_argument(
        "--norm",
        default=None,
        help="Normalization: 'two_slope' centers at zero; None for linear."
    )
    return parser.parse_args()

def get_extractor_class(extractor_name):
    """
    Get the class of the extractor based on its name.
    """
    if extractor_name in feature_extractors.__dict__:
        return feature_extractors.__dict__[extractor_name]
    elif extractor_name in slide_level_predictors.__dict__:
        return slide_level_predictors.__dict__[extractor_name]
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")


def process_slide(args, slide_path, output_dir):
    logging.info(f"\nProcessing slide: {slide_path}")
    base = os.path.splitext(os.path.basename(slide_path))[0]
    with open(os.path.join(args.model_dir, "mil_params.json"), "r") as f:
        # Load model parameters
        try:
            model_dict = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error loading JSON: {e}")
            sys.exit(1)
    logging.info(f"Model parameters: {model_dict}")

    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error loading YAML: {e}")
            sys.exit(1)

    #Login using hf_token
    token = config['hf_key']
    if token:
        from huggingface_hub import login
        login(token=token)
    logging.info(f"Logged in to Hugging Face with token: {token}")

    #Set pretrained weights directory
    WEIGHTS_DIR = config['weights_dir']
    # Set environment variables
    os.environ['TORCH_HOME'] = WEIGHTS_DIR
    os.environ['HF_HOME'] = WEIGHTS_DIR
    os.environ['XDG_CACHE_HOME'] = WEIGHTS_DIR
    os.environ['HF_DATASETS_CACHE'] = WEIGHTS_DIR
    os.environ['WEIGHTS_DIR'] = WEIGHTS_DIR

    logging.info(f"Set environment variables for pretrained weights directory: {WEIGHTS_DIR}")
    
    extractor_name = model_dict['bags_extractor']['extractor']['class']
    extractor_class = get_extractor_class(extractor_name)

    normalizer = model_dict['bags_extractor']['normalizer']['method']
    logging.info(f"Normalizer: {normalizer}")

    mil_model = model_dict['params']['model']
    logging.info(f"MIL model: {mil_model}")

    tile_px, tile_um = model_dict['bags_extractor']['tile_px'], model_dict['bags_extractor']['tile_um']

    heatmap_kwargs = {
        "interpolation": args.interpolation,
        "cmap": args.cmap,
        "norm": args.norm,
    }
    # 1) Run inference (and get 2D attention heatmap if requested)
    preds, attn_2d = predict_slide(
        model=args.model_dir,
        slide=slide_path,
        attention=args.heatmap,
        heatmap_kwargs=heatmap_kwargs
    )

    # 2) Save predictions
    pred_file = os.path.join(output_dir, f"{base}_preds.npy")
    np.save(pred_file, preds)
    logging.info(f"Saved predictions: {pred_file}, shape: {preds.shape}")

    # Save uncertainty if requested
    if args.uncertainty:
        uncertainty = calculate_uncertainty(preds)
        uncertainty_file = os.path.join(output_dir, f"{base}_uncertainty.npy")
        np.save(uncertainty_file, uncertainty)
        logging.info(f"Saved uncertainty: {uncertainty_file}, shape: {uncertainty.shape}")


def calculate_uncertainty(preds, uncertainty="entropy"):
    """
    Calculate uncertainty based on the predictions.
    """
    if uncertainty == "entropy":
        # Example: using entropy as a measure of uncertainty
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    elif uncertainty == "variance":
        # Example: using variance as a measure of uncertainty
        uncertainty = np.var(preds, axis=1)
    else:
        raise ValueError(f"Unknown uncertainty method: {uncertainty}")
    return uncertainty

def main():
    args = parse_args()

    if not os.path.isdir(args.model_dir):
        logging.error(f"Error: model_dir '{args.model_dir}' does not exist.")
        sys.exit(1)

    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)

    # Collect slides
    if args.slide:
        if not os.path.isfile(args.slide):
            logging.error(f"Error: slide '{args.slide}' not found.")
            sys.exit(1)
        slides = [args.slide]
    else:
        if not os.path.isdir(args.slide_dir):
            logging.error(f"Error: slide_dir '{args.slide_dir}' not found.")
            sys.exit(1)
        slides = [
            os.path.join(args.slide_dir, f)
            for f in os.listdir(args.slide_dir)
            if os.path.isfile(os.path.join(args.slide_dir, f))
        ]

    if not slides:
        logging.error("No slides to process.")
        sys.exit(1)

    for slide_path in slides:
        process_slide(args, slide_path, output_dir)


if __name__ == "__main__":
    main()
