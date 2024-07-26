FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev gfortran libatlas-base-dev cmake

RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN python3 -m pip install -U pip

RUN python3 -m pip install 'setuptools==60.5.0' 'Cython==0.29.26' 'typing_extensions==4.0.1'
RUN python3 -m pip install 'numpy==1.21.5'
RUN python3 -m pip install 'pythran' 'pybind11'
RUN python3 -m pip install 'scipy==1.7.3'
RUN python3 -m pip install 'luigi>=3.0.0' 'scikit-learn==1.0.2' 'pyarrow==6.0.1' 'pyspark==3.4.2' 'tqdm==4.62.3' \
                           'pandas==1.3.5' 'duckdb' 'pytest' 'pylint' 'coverage' 'pyhocon'
RUN python3 -m pip install 'torch==1.12.1' 'pytorch-lightning==1.6.5' 'torchmetrics==0.9.2' \
                           'hydra-core>=1.1.2' 'hydra-optuna-sweeper>=1.2.0' 'tensorboard==2.3.0' \
                           'omegaconf' 'transformers' 'lightgbm' 'wandb'

RUN python3 -m pip cache purge
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
