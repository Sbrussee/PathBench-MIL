.. PathBench documentation master file, created by
   sphinx-quickstart on Fri Aug 6 14:36:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
Welcome to PathBench's Documentation
=====================================

.. raw:: html

    <div style="text-align: center;">
    <img src="thumbnail_PathBench-logo-horizontaal.png" alt="PathBench" width="800" height="200">
    </div>

---

Multiple Instance Learning (MIL) has emerged as the predominant modeling paradigm for computational histopathology models. These MIL models employ feature extractors to extract patch embeddings, which are then aggregated to calculate a final slide-level prediction. Lately, foundation models have become the state-of-the-art feature extractors for obtaining patch embeddings. 

The use of MIL models, especially for individuals without an engineering background, has been hindered by the vast design space associated with constructing MIL models. Here, we present **PathBench**, an open-source, extendable framework for benchmarking MIL deep learning pipelines for histopathology, which integrates a wide variety of feature extractor and MIL aggregation models.

PathBench automates the process of finding well-performing MIL pipelines for pathology tasks. It allows for the principled construction of ensemble models based on semantic feature extraction similarity and includes an interactive visualization app. By introducing PathBench, we significantly reduce the time and complexity associated with constructing MIL models for histopathology, democratizing the use of AI in histopathology, especially for medical professionals.

PathBench is being developed at the Leiden University Medical Center: Department of Pathology.

- **Lead Developer**: Siemen Brussee
- **Developers**: 

Explore the documentation to learn more about how to install, configure, and extend PathBench for your specific needs.

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   benchmarking_mode
   optimization_mode
   visualization_application
   ensemble_model_creation
   extending_pathbench
   feature_extractors
   mil_aggregators
   modules
