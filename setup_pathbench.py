import os
import subprocess
import sys
import venv

def create_virtualenv(env_name):
    venv.create(env_name, with_pip=True)
    print(f"Created virtual environment in {env_name}")

def install_wheel_and_versioneer(env_name):
    # Determine the paths for the virtual environment
    if os.name == 'nt':
        bin_path = os.path.join(env_name, 'Scripts')
    else:
        bin_path = os.path.join(env_name, 'bin')
    
    python_executable = os.path.join(bin_path, 'python')
    pip_executable = os.path.join(bin_path, 'pip')

    # Install wheel and versioneer
    subprocess.check_call([pip_executable, 'install', 'wheel', 'versioneer'])

def run_setup_py(env_name):
    if os.name == 'nt':
        bin_path = os.path.join(env_name, 'Scripts')
    else:
        bin_path = os.path.join(env_name, 'bin')
    
    python_executable = os.path.join(bin_path, 'python')

    subprocess.check_call([python_executable, 'setup.py', 'install'])

def main():
    env_name = 'pathbench_env'

    create_virtualenv(env_name)
    install_wheel_and_versioneer(env_name)
    run_setup_py(env_name)

if __name__ == "__main__":
    main()