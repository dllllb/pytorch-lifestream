import torch
import torch.multiprocessing as mp
import numpy as np
from get_model import get_coles_module, get_static_multicoles_module
from get_data import get_synthetic_sup_datamodule, get_age_pred_sup_datamodule
from pyhocon import ConfigFactory
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier


def load_monomodel(model_path, conf_path='./config.hocon'):
    conf = ConfigFactory.parse_file(conf_path)
    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = conf.get('hsize')
    module = get_coles_module(trx_conf, input_size, hsize)
    module.seq_encoder.load_state_dict(torch.load(model_path))
    return module


def load_multimodel(first_model_path, second_model_path, conf_path='./config.hocon'):
    conf = ConfigFactory.parse_file(conf_path)
    trx_conf = conf.get('trx_conf')
    input_size = conf.get('input_size')
    hsize = conf.get('hsize')
    clf_hsize = conf.get('clf_hsize', 64)
    module = get_static_multicoles_module(trx_conf, input_size, 1., hsize, clf_hsize, first_model_path)
    module.seq_encoder.load_state_dict(torch.load(second_model_path))
    return module


def predict_on_dataloader(model, dataloader, gpu_n):
    xx, yy = list(), list()
    dl = iter(dataloader)
    for batch in dl:

        with torch.no_grad():
            x, y = batch
            yy.append(y.numpy())

            x_list = list()
            if hasattr(model, 'trained_models'):
                if model.trained_models is not None:
                    x_list.extend([m(x.to('cuda:' + str(gpu_n))).detach().cpu().numpy() for m in model.trained_models])
            x_list.append(model(x.to('cuda:' + str(gpu_n))).detach().cpu().numpy())

            x = np.concatenate(x_list[0:], axis=-1)
            xx.append(x)

    xx = np.concatenate(xx, axis=0)
    yy = np.concatenate(yy, axis=0)
    return xx, yy


def solve_downstream(xx, yy, test_xx, test_yy, metric_f, conf_path='./config.hocon'):
    scores = list()
    conf = ConfigFactory.parse_file(conf_path)
    lgbm_conf = conf.get('lgbm_conf')
    for gbm_i in range(5):
        lgbm_conf['random_state'] = 42 + gbm_i
        clf = LGBMClassifier(**lgbm_conf)
        clf.fit(xx, yy)
        pred_y = clf.predict_proba(test_xx)[:, 1]
        test_score = metric_f(test_yy, pred_y)
        scores.append(test_score)
    return scores


def inference(mode, models_info, conf_path='./config.hocon'):
    conf = ConfigFactory.parse_file(conf_path)
    dataset = conf.get('dataset')
    gpu_n = conf.get('gpu_n')
    if dataset == 'synthetic':
        dataf = get_synthetic_sup_datamodule
        metric = roc_auc_score
    elif dataset == 'age_pred':
        dataf = get_age_pred_sup_datamodule
        metric = accuracy_score

    if mode == 'mono':
        model_loader = load_monomodel
    else:
        model_loader = load_multimodel


    def predict_on_fold(task_info):
        model_loading_info, fold_i = task_info
        sup_data = dataf(fold_i)
        model = model_loader(*model_loading_info)

        train_dl = sup_data.train_dataloader()
        train_xx, train_yy = predict_on_dataloader(model, train_dl, gpu_n)

        test_dl = sup_data.test_dataloader()
        test_xx, test_yy = predict_on_dataloader(model, test_dl, gpu_n)

        metric_scores = solve_downstream(train_xx, train_yy, test_xx, test_yy, metric, conf_path=conf_path)
        return fold_i, metric_scores


    with mp.Pool(5) as p:
        scores = p.map(predict_on_fold, models_info, chunksize=1)
    return scores
