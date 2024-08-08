Ensemble Model Creation
=======================

PathBench includes a pipeline to calculate the semantic feature similarity between feature bags, which can then be used to build model ensembles.

To create an ensemble model:

1. Run the feature similarity calculation pipeline.
2. Use the output to create an ensemble model.

Example usage:

```bash
python ensemble_model_creation.py path_to_feature_bags