import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from get_data import get_age_pred_coles_datamodule, get_synthetic_coles_datamodule, get_alpha_battle_coles_datamodule
from get_model import get_coles_module, get_static_multicoles_module
from get_paths import add_next_ind

from pyhocon import ConfigFactory
from datetime import datetime


def train_coles_model_folder(fold_i, exp_name, dataf, trx_conf, input_size, hsize,
                             path_to_logs, path_to_chkp, is_mono, gpu_n, max_epoch, debug):
    datamodule = dataf(fold_i)
    module = get_coles_module(trx_conf, input_size, hsize)
    writer = TensorBoardLogger(save_dir=path_to_logs, name=is_mono + '_fold_' + str(fold_i))
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
    tb_name = exp_name + '/' + is_mono + '_fold_' + str(fold_i) + '/' + str(logger_version)
    trainer.fit(module, datamodule)
    name = '_'.join([is_mono, str(fold_i)])
    name = add_next_ind(path_to_chkp, name) + '.pth'
    model_save_path = os.path.join(path_to_chkp, name)
    torch.save(module.seq_encoder.state_dict(), model_save_path)
    return fold_i, tb_name, model_save_path


def train_coles_model(exp_name, path_to_chkp, path_to_logs,
                      fold_i, gpu_n, monomodel=True, conf_path='./config.hocon', debug=False):
    conf = ConfigFactory.parse_file(conf_path)
    dataset = conf.get('dataset')
    assert dataset in ['synthetic', 'age_pred', 'alpha_battle'], 'invalid dataset'
    if dataset == 'synthetic':
        dataf = get_synthetic_coles_datamodule
    elif dataset == 'age_pred':
        dataf = get_age_pred_coles_datamodule
    elif dataset == 'alpha_battle':
        dataf = get_alpha_battle_coles_datamodule

    #time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = conf.get('hsize')
    max_epoch = conf.get('max_epoch', 50)

    if monomodel:
        is_mono = 'full'
    else:
        hsize = int(hsize / 2)
        is_mono = 'first_half'

    paths_to_model = train_coles_model_folder(fold_i, exp_name, dataf, trx_conf, input_size, hsize,
                                              path_to_logs, path_to_chkp, is_mono, gpu_n, max_epoch, debug)
    return paths_to_model


def train_model_folder(fold_i, first_model_path, exp_name, dataf, trx_conf, input_size, hsize,
                       clf_hsize, path_to_logs, path_to_chkp, embed_coef, gpu_n, max_epoch, debug):
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
    name = '_'.join(['second_half', str(fold_i)])
    name = add_next_ind(path_to_chkp, name) + '.pth'
    model_save_path = os.path.join(path_to_chkp, name)
    torch.save(module.seq_encoder.state_dict(), model_save_path)
    return fold_i, tb_name, model_save_path


def train_multicoles_model(path_to_logs, path_to_chkp, fold_i, first_model_path, gpu_n,
                           conf_path='./config.hocon', debug=False):
    conf = ConfigFactory.parse_file(conf_path)
    dataset = conf.get('dataset')
    assert dataset in ['synthetic', 'age_pred', 'alpha_battle'], 'invalid dataset'
    if dataset == 'synthetic':
        dataf = get_synthetic_coles_datamodule
    elif dataset == 'age_pred':
        dataf = get_age_pred_coles_datamodule
    elif dataset == 'alpha_battle':
        dataf = get_alpha_battle_coles_datamodule

    exp_name = conf.get('exp_name', 'default_name')
    #time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = int(conf.get('hsize')/2)
    clf_hsize = conf.get('clf_hsize', 64)
    max_epoch = conf.get('max_epoch', 50)
    embed_coef = conf.get('embed_coef', 0.1)

    paths_to_models = train_model_folder(fold_i, first_model_path, exp_name, dataf, trx_conf, input_size, hsize,
                                         clf_hsize, path_to_logs, path_to_chkp, embed_coef, gpu_n, max_epoch, debug)
    return paths_to_models
