#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='pytorch-lifestream',
    version='0.6.0',
    author='',
    author_email='',
    description='Lifestream data analysis with PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'duckdb',
        'hydra-core>=1.1.2',
        'numpy>=1.21.5',
        'omegaconf',
        'pandas>=1.3.5',
        'pyarrow>=6.0.1',
        'pytorch-lightning>=1.6.0',
        'scikit-learn>=1.0.2',
        'torch>=1.12.0',
        'torchmetrics>=0.9.0',
        'transformers',
        'dask',
        'pymonad',
        'spacy==3.7.4',
        'fedcore>=0.0.4.5'
    ],
)
