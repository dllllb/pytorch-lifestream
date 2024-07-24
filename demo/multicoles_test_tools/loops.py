import os
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from get_data import get_age_pred_coles_datamodule, get_synthetic_coles_datamodule
from get_model import get_coles_module, get_static_multicoles_module
from get_paths import create_experiment_folder
from pyhocon import ConfigFactory
from datetime import datetime


def train_coles_model(monomodel=True, conf_path='./config.hocon', debug=False):
    conf = ConfigFactory.parse_file(conf_path)
    dataset = conf.get('dataset')
    assert dataset in ['synthetic', 'age_pred'], 'invalid dataset'
    if dataset == 'synthetic':
        dataf = get_synthetic_coles_datamodule
    elif dataset == 'age_pred':
        dataf = get_age_pred_coles_datamodule

    exp_name = conf.get('exp_name', 'default_name')
    path_to_exp, path_to_chkp, path_to_logs = create_experiment_folder(exp_name)
    #time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = conf.get('hsize')
    gpu_n = conf.get('gpu_n')
    max_epoch = conf.get('max_epoch', 50)

    if monomodel:
        hsize = hsize * 2
        is_mono = 'first_half'
    else:
        is_mono = 'full'


    def train_model_folder(fold_i):
        datamodule = dataf(fold_i)
        module = get_coles_module(trx_conf, input_size, hsize)
        writer = TensorBoardLogger(save_dir=path_to_logs, name=is_mono+'_fold_'+str(fold_i))
        trainer = pl.Trainer(gpus=[gpu_n],
                             max_epochs=max_epoch,
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             gradient_clip_val=0.5,
                             gradient_clip_algorithm="value",
                             track_grad_norm=2,
                             logger=writer,
                             fast_dev_run=debug)
        logger_version = trainer.logger.version
        tb_name = exp_name + '/' + is_mono+'_fold_'+str(fold_i) + '/' + str(logger_version)
        trainer.fit(module, datamodule)
        model_save_path = os.path.join(path_to_chkp, '_'.join([is_mono, str(fold_i)])+'.pth')
        torch.save(module.seq_encoder.state_dict(), model_save_path)
        return fold_i, tb_name, model_save_path


    with mp.Pool(5) as p:
        paths_to_models = p.map(train_model_folder, [i for i in range(5)], chunksize=1)
    return paths_to_models


def train_multicoles_model(first_models, embed_coef, conf_path='./config.hocon', debug=False):
    conf = ConfigFactory.parse_file(conf_path)
    dataset = conf.get('dataset')
    assert dataset in ['synthetic', 'age_pred'], 'invalid dataset'
    if dataset == 'synthetic':
        dataf = get_synthetic_coles_datamodule
    elif dataset == 'age_pred':
        dataf = get_age_pred_coles_datamodule

    exp_name = conf.get('exp_name', 'default_name')
    path_to_exp, path_to_chkp, path_to_logs = create_experiment_folder(exp_name)
    #time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = conf.get('hsize')
    clf_hsize = conf.get('clf_hsize', 64)
    gpu_n = conf.get('gpu_n')
    max_epoch = conf.get('max_epoch', 50)


    def train_model_folder(fold_info):
        fold_i, _, first_model_path = fold_info
        datamodule = dataf(fold_i)
        module = get_static_multicoles_module(trx_conf, input_size, embed_coef, hsize, clf_hsize, first_model_path)
        writer = TensorBoardLogger(save_dir=path_to_logs, name='second_half_fold_'+str(fold_i))
        trainer = pl.Trainer(gpus=[gpu_n],
                             max_epochs=max_epoch,
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             track_grad_norm=2,
                             logger=writer,
                             fast_dev_run=debug)
        logger_version = trainer.logger.version
        tb_name = exp_name + '/' + 'second_half_fold_' + str(fold_i) + '/' + str(logger_version)
        trainer.fit(module, datamodule)
        model_save_path = os.path.join(path_to_chkp, '_'.join(['second_half', str(fold_i)])+'.pth')
        torch.save(module.seq_encoder.state_dict(), model_save_path)
        return fold_i, tb_name, model_save_path


    with mp.Pool(5) as p:
        paths_to_models = p.map(train_model_folder, first_models, chunksize=1)
    return paths_to_models
