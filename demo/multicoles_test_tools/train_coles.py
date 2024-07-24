import sys
import numpy as np
from loops import train_coles_model
from downstream_test import inference
from get_paths import write_results
from pyhocon import ConfigFactory


def main():
    conf_path = str(sys.argv[3])
    debug = True if str(sys.argv[4]) == 'debug' else False
    gpu_n = int(sys.argv[2])
    fold_i = int(sys.argv[1])
    if conf_path == 'def':
        conf_path = './config.hocon'

    conf = ConfigFactory.parse_file(conf_path)
    exp_name = conf.get('exp_name', 'default_name')

    # list of (fold_i, tb_name, model_save_path)
    path_to_model = train_coles_model(fold_i=fold_i, gpu_n=gpu_n, monomodel=True, conf_path=conf_path, debug=debug)
    task_info = [path_to_model[2], True], fold_i
    # list of (fold_i, metric_scores)
    score = inference(mode='mono', task_info=task_info, gpu_n=gpu_n, conf_path=conf_path)

    result = {
        'fold_i': path_to_model[0],
        'tb_name': path_to_model[1],
        'model_path': path_to_model[2],
        'scores': ','.join([str(np.round(float(x), 5)) for x in score[1]]),
        'mean_score': str(np.mean(score[1])),
        'net_type': 'mono'
    }
    write_results(exp_name, result)


if __name__ == '__main__':
    main()
