#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='pytorch-lifestream',
    version='0.2.0',
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
    python_requires='>=3.8',
    install_requires=[
        'pytorch-lightning==1.5.*',
        'pyarrow==7.*',
        'transformers==4.*'
    ],
)
