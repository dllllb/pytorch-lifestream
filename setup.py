#!/usr/bin/env python

from distutils.core import setup

setup(
    name='pytorch-lifestream',
    version='0.1.1',
    description='Llifestream data anaysis with PyTorch',
    author='Dmitri Babaev',
    author_email='dmitri.babaev@gmail.com',
    install_requires=[
        'torch>=1.1.0',
        'pytorch-ignite>=0.2.1',
        'scikit-learn>=0.21.2',
	'numpy>=1.16.4',
	'pandas>=0.24.2',
        'scipy>=1.3.0',
        'tqdm>=4.32.2',
        'tensorboard>=2.1.0']
)
