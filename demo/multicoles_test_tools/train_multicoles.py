import sys
import os
import numpy as np
from loops import train_multicoles_model
from downstream_test import inference
from get_paths import write_results
from pyhocon import ConfigFactory
from get_paths import create_experiment_folder, save_config_copy
from train_coles import train_mono


def main():
    conf_path = str(sys.argv[3])
    debug = True if str(sys.argv[4]) == 'debug' else False
    gpu_n = int(sys.argv[2])
    fold_i = int(sys.argv[1])
    if conf_path == 'def':
        conf_path = './config.hocon'

    conf = ConfigFactory.parse_file(conf_path)
    exp_name = conf.get('exp_name', 'default_name')
    path_to_exp, path_to_chkp, path_to_logs = create_experiment_folder(exp_name)
    if fold_i == 0:
        save_config_copy(conf_path, exp_name)
    first_model_path = conf.get('first_model_path', None)
    remake_first_model = conf.get('remake_first_model', False)
    indep = conf.get('indep_mode', False)
    nettype = 'indep' if indep else 'multi'

    if first_model_path is None:
        first_model_path = os.path.join(path_to_chkp, '_'.join(['first_half', str(fold_i)])+'.pth')
        if not os.path.isfile(first_model_path) or remake_first_model:
            train_mono(monomodel=False)

    embed_coef = conf.get('embed_coef', 0.1)
    assert type(embed_coef) in [float, list, tuple]
    if type(embed_coef) is float:
        embed_coef = [embed_coef]

    for coef in embed_coef:
        # list of (fold_i, tb_name, model_save_path)
        path_to_model = train_multicoles_model(path_to_logs=path_to_logs, path_to_chkp=path_to_chkp,
                                               fold_i=fold_i, first_model_path=first_model_path, embed_coef=coef,
                                               gpu_n=gpu_n, conf_path=conf_path, debug=debug)
        task_info = {'first_model_path': first_model_path, 'second_model_path': path_to_model[2]}, fold_i
        # list of (fold_i, metric_scores)
        score = inference(mode='multi', task_info=task_info, gpu_n=gpu_n, conf_path=conf_path, debug=debug)

        result = {
            'fold_i': str(path_to_model[0]),
            'tb_name': path_to_model[1],
            'model_path': path_to_model[2],
            'scores': ','.join([str(np.round(float(x), 5)) for x in score[1]]),
            'mean_score': str(np.mean(score[1])),
            'net_type': nettype,
            'coef': str(coef),
            'ddim': str(conf.get('clf_hsize', 64))
        }
        write_results(exp_name, result)


if __name__ == '__main__':
    main()
