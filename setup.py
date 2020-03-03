#!/usr/bin/env python

from distutils.core import setup

setup(
    name='pytorch-lifestream',
    version='0.1.1',
    description='Llifestream data anaysis with PyTorch',
    author='Dmitri Babaev',
    author_email='dmitri.babaev@gmail.com',
    install_requires=['torch>=1.1.0', 'pytorch-ignite>=0.2.1']
)
