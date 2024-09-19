from setuptools import setup, find_packages
import os

#Read requirements
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='pathbench',
    version='0.1.0',
    description='PathBench-MIL: A flexible, comprehensive benchmarking / AutoML framework for MIL models in Histopathology',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='S Brussee',
    author_email='s.brussee@lumc.nl',
    url='https://github.com/sbrussee/PathBench-MIL',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=read_requirements(),
)
