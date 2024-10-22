Installation
============

Prerequisites
-------------

- Python 3.8
- Git

Steps to Install PathBench and SlideFlow Fork
---------------------------------------------

1. **Clone the Repository:**

.. code-block:: bash

    git clone --recursive_submodules https://github.com/sbrussee/PathBench-MIL.git
    cd PathBench-MIL

2. **Run `setup_pathbench.py`:**

    Run the existing script to set up the virtual environment and install necessary tools.

.. code-block:: bash

    python setup_pathbench.py

    This script will:
    - Create a virtual environment named `pathbench_env`.
    - Upgrade `pip` and install `setuptools`, `wheel`, and `versioneer`.

3. **Activate the Virtual Environment:**

    - macOS/Linux:

.. code-block:: bash

        source pathbench_env/bin/activate

    - Windows:

.. code-block:: bash

        pathbench_env\Scripts\activate

5. **Install `slideflow_fork` Package:**

    Navigate to the `slideflow_fork` directory and install it:

.. code-block:: bash

    cd ../slideflow_fork
    pip install -e .

    Or, if you do not need to modify the code:

.. code-block:: bash

    pip install .


5. **Install `pathbench` Package:**

    After installing slideflow, install the `pathbench` package.

.. code-block:: bash

    pip install -e .

    Or, if you do not need to modify the code:

.. code-block:: bash

    pip install .

5. **Install `slideflow_fork` Package:**

    Navigate to the `slideflow_fork` directory and install it:

.. code-block:: bash

    cd ../slideflow_fork
    pip install -e .

    Or, if you do not need to modify the code:

.. code-block:: bash

    pip install .
