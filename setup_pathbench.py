import os
import subprocess
import sys
import venv

def create_virtualenv(env_name):
    venv.create(env_name, with_pip=True)

def install_base_packages(env_name):
    # Determine the paths for the virtual environment
    if os.name == 'nt':
        bin_path = os.path.join(env_name, 'Scripts')
    else:
        bin_path = os.path.join(env_name, 'bin')
    
    pip_executable = os.path.join(bin_path, 'pip')

    # Install wheel, versioneer, cython, and ruamel
    subprocess.check_call([pip_executable, 'install', 'wheel', 'versioneer', 'cython', 'ruamel.yaml'])

def run_setup_py(env_name):
    # Activate the virtual environment and run setup.py
    if os.name == 'nt':
        activate_script = os.path.join(env_name, 'Scripts', 'activate')
        python_executable = os.path.join(env_name, 'Scripts', 'python')
    else:
        activate_script = os.path.join(env_name, 'bin', 'activate')
        python_executable = os.path.join(env_name, 'bin', 'python')

    # Source the virtual environment activation script and run setup.py
    command = f'source {activate_script} && {python_executable} setup.py install'
    subprocess.check_call(command, shell=True, executable='/bin/bash')

def main():
    env_name = 'pathbench_env'

    create_virtualenv(env_name)
    print(f"Created virtual environment in {env_name}")
    install_base_packages(env_name)
    print("Installed base packages")
    run_setup_py(env_name)
    print("Installed PathBench")

if __name__ == "__main__":
    main()