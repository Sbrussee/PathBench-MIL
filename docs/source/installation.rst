Installation
============

## Prerequisites

- Python 3.8
- Git

## Steps to Install PathBench and SlideFlow Fork

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/sbrussee/PathBench.git
    cd PathBench
    ```

2. **Run `setup_pathbench.py`:**

    Run the existing script to set up the virtual environment and install necessary tools.

    ```bash
    python setup_pathbench.py
    ```

    This script will:
    - Create a virtual environment named `pathbench_env`.
    - Upgrade `pip` and install `setuptools`, `wheel`, and `versioneer`.

3. **Activate the Virtual Environment:**

    - macOS/Linux:
        ```bash
        source pathbench_env/bin/activate
        ```
    - Windows:
        ```bash
        pathbench_env\Scripts\activate
        ```

4. **Install `pathbench` Package:**

    After activating the virtual environment, install the `pathbench` package.

    ```bash
    pip install -e .
    ```

    Or, if you do not need to modify the code:

    ```bash
    pip install .
    ```

5. **Install `slideflow_fork` Package:**

    Navigate to the `slideflow_fork` directory and install it:

    ```bash
    cd ../slideflow_fork
    pip install -e .
    ```

    Or, if you do not need to modify the code:

    ```bash
    pip install .
    ```