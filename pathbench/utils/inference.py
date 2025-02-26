#!/usr/bin/env python
import argparse
import os
import sys
import json
import numpy as np
import logging
import slideflow.mil as mil

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using a saved MIL model directory on a slide or all slides in a directory, and save results."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved model directory (should contain mil_params.json, models/, etc.)"
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
        "--attention",
        action="store_true",
        help="If provided, saves attention scores along with predictions."
    )
    return parser.parse_args()

def process_slide(model_dir, slide_path, output_dir, attention=False):
    logging.info(f"\nProcessing slide: {slide_path}")
    try:
        # Run prediction using the MIL model
        predictions, attn = mil.predict_slide(
            model=model_dir,
            slide=slide_path,
            attention=attention
        )

        #Print the predictions
        logging.info(f"Predictions:\n{predictions}")
        
        # Determine the base name for output files
        base_name = os.path.splitext(os.path.basename(slide_path))[0]
        pred_file = os.path.join(output_dir, f"{base_name}_predictions.npy")
        np.save(pred_file, predictions)
        logging.info(f"Saved predictions to: {pred_file}")
        
        # If attention is requested and returned, save it
        if attention:
            if attn is not None:
                attn_file = os.path.join(output_dir, f"{base_name}_attention.npy")
                np.save(attn_file, attn)
                logging.info(f"Saved attention scores to: {attn_file}")
            else:
                logging.warning("No attention scores returned.")
    except Exception as e:
        logging.warning(f"Error processing slide {slide_path}: {e}")

def main():
    args = parse_args()

    # Verify that the model directory exists
    if not os.path.exists(args.model_dir):
        logging.warning(f"Error: model_dir '{args.model_dir}' does not exist.")
        sys.exit(1)

    # Prepare the output directory for inference results
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)

    # Collect slide paths based on provided arguments
    slide_paths = []
    if args.slide:
        if not os.path.isfile(args.slide):
            logging.warning(f"Error: slide '{args.slide}' is not a valid file.")
            sys.exit(1)
        slide_paths.append(args.slide)
    elif args.slide_dir:
        if not os.path.isdir(args.slide_dir):
            logging.warning(f"Error: slide_dir '{args.slide_dir}' is not a valid directory.")
            sys.exit(1)
        # Optionally, filter by slide file extension (e.g., .svs)
        for file in os.listdir(args.slide_dir):
            file_path = os.path.join(args.slide_dir, file)
            if os.path.isfile(file_path):
                slide_paths.append(file_path)

    if not slide_paths:
        logging.warning("No slides found to process.")
        sys.exit(1)

    # Process each slide and save the inference results
    for slide_path in slide_paths:
        process_slide(args.model_dir, slide_path, output_dir, attention=args.attention)

if __name__ == "__main__":
    main()
