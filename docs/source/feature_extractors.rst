Feature Extractors
==================

PathBench supports a wide range of different feature extractors, including SOTA foundation models for pathology. Most of these models are automatically downloaded by PathBench; however, some models require a Hugging Face account key to access the model (labeled 'Gated' in the feature extraction table) or require manually downloading the model weights (labeled 'Manual' in the extraction table). For each of the models, a link to the publication is given, and for the manual/gated models, the link for downloading the models or gaining model access are also provided.

.. list-table::
   :header-rows: 1

   * - Feature Extractor
     - Acquisition
     - Link
   * - ImageNet-ResNet50
     - Automatic
     - NA
   * - CTransPath
     - Automatic
     - `Link <https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file>`_
   * - MoCoV3-TransPath
     - Automatic
     - `Link <https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file>`_
   * - HistoSSL
     - Automatic
     - `Link <https://github.com/owkin/HistoSSLscaling>`_
   * - RetCCL
     - Automatic
     - `Link <https://github.com/Xiyue-Wang/RetCCL>`_
   * - PLIP
     - Automatic
     - `Link <https://github.com/PathologyFoundation/plip?tab=readme-ov-file>`_
   * - Lunit DINO
     - Automatic
     - `Link <https://github.com/lunit-io/benchmark-ssl-pathology>`_
   * - Lunit SwAV
     - Automatic
     - `Link <https://github.com/lunit-io/benchmark-ssl-pathology>`_
   * - Lunit Barlow Twins
     - Automatic
     - `Link <https://github.com/lunit-io/benchmark-ssl-pathology>`_
   * - Lunit MocoV2
     - Automatic
     - `Link <https://github.com/lunit-io/benchmark-ssl-pathology>`_
   * - Phikon
     - Automatic
     - `Link <https://huggingface.co/owkin/phikon>`_
   * - PathoDuet-HE
     - Manual
     - `Link <https://github.com/openmedlab/PathoDuet>`_ `Weights <https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM>`_
   * - PathoDuet-IHC
     - Manual
     - `Link <https://github.com/openmedlab/PathoDuet>`_ `Weights <https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM>`_
   * - Virchow
     - Gated
     - `Link <https://huggingface.co/paige-ai/Virchow>`_
   * - Hibou-B
     - Automatic
     - `Link <https://huggingface.co/histai/hibou-b>`_
   * - UNI
     - Gated
     - `Link <https://huggingface.co/MahmoodLab/UNI>`_
   * - Prov-GigaPath
     - Gated
     - `Link <https://huggingface.co/prov-gigapath/prov-gigapath>`_
   * - Kaiko-S8
     - Automatic
     - `Link <https://github.com/kaiko-ai/towards_large_pathology_fms>`_
   * - Kaiko-S16
     - Automatic
     - `Link <https://github.com/kaiko-ai/towards_large_pathology_fms>`_
   * - Kaiko-B8
     - Automatic
     - `Link <https://github.com/kaiko-ai/towards_large_pathology_fms>`_
   * - Kaiko-B16
     - Automatic
     - `Link <https://github.com/kaiko-ai/towards_large_pathology_fms>`_
   * - Kaiko-L14
     - Automatic
     - `Link <https://github.com/kaiko-ai/towards_large_pathology_fms>`_
   * - H-Optimus-0
     - Automatic
     - `Link <https://huggingface.co/bioptimus/H-optimus-0>`_