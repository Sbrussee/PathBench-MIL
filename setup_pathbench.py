import subprocess
import os
import venv

def create_virtualenv(env_name):
    venv.create(env_name, with_pip=True)
    print(f"Created virtual environment in {env_name}")

def upgrade_pip_and_install_tools(env_name):
    bin_path = os.path.join(env_name, 'Scripts' if os.name == 'nt' else 'bin')
    pip_executable = os.path.join(bin_path, 'pip')
    
    try:
        subprocess.check_call([pip_executable, 'install', '--upgrade', 'pip'])
        print("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade pip: {e}")

    try:
        subprocess.check_call([pip_executable, 'install', 'setuptools', 'wheel', 'versioneer'])
        print("Setuptools, Wheel, and Versioneer installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install setuptools, wheel, and versioneer: {e}")

def main():
    env_name = 'pathbench_env'
    create_virtualenv(env_name)
    upgrade_pip_and_install_tools(env_name)
    print(f"To activate the virtual environment, use:\nsource {env_name}/bin/activate (on macOS/Linux) or {env_name}\\Scripts\\activate (on Windows)")
    print("After activating the virtual environment, run `pip install -e .` to install the package.")

if __name__ == "__main__":
    main()