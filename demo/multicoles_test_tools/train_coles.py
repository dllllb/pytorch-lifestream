import sys
import numpy as np
from loops import train_coles_model
from downstream_test import inference
from get_paths import write_results
from pyhocon import ConfigFactory
from get_paths import create_experiment_folder, save_config_copy


def train_mono(monomodel=True):
    conf_path = str(sys.argv[3])
    debug = True if str(sys.argv[4]) == 'debug' else False
    gpu_n = int(sys.argv[2])
    fold_i = int(sys.argv[1])
    nettype = 'mono' if monomodel else 'first'
    if conf_path == 'def':
        conf_path = './config.hocon'

    conf = ConfigFactory.parse_file(conf_path)
    exp_name = conf.get('exp_name', 'default_name')
    path_to_exp, path_to_chkp, path_to_logs = create_experiment_folder(exp_name)
    if fold_i == 0:
        save_config_copy(conf_path, exp_name)

    # list of (fold_i, tb_name, model_save_path)
    path_to_model = train_coles_model(exp_name=exp_name, path_to_chkp=path_to_chkp, path_to_logs=path_to_logs,
                                      fold_i=fold_i, gpu_n=gpu_n, monomodel=monomodel, conf_path=conf_path, debug=debug)
    task_info = {'model_path': path_to_model[2], 'mono': monomodel}, fold_i
    # list of (fold_i, metric_scores)
    score = inference(mode='mono', task_info=task_info, gpu_n=gpu_n, conf_path=conf_path, debug=debug)

    result = {
        'fold_i': str(path_to_model[0]),
        'tb_name': path_to_model[1],
        'model_path': path_to_model[2],
        'scores': ','.join([str(np.round(float(x), 5)) for x in score[1]]),
        'mean_score': str(np.mean(score[1])),
        'net_type': nettype,
        'coef': 'None',
        'ddim': 'None'
    }
    write_results(exp_name, result)


if __name__ == '__main__':
    train_mono(monomodel=True)
