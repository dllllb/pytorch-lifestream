import sys
import numpy as np
import torch.multiprocessing as mp
from .loops import train_coles_model
from downstream_test import inference
from get_paths import write_results
from pyhocon import ConfigFactory


mp.set_sharing_strategy('spawn')


def main():
    conf_path = str(sys.argv[1])
    debug = True if str(sys.argv[2]) == 'debug' else False
    if conf_path == 'def':
        conf_path = './config.hocon'

    conf = ConfigFactory.parse_file(conf_path)
    exp_name = conf.get('exp_name', 'default_name')

    # list of (fold_i, tb_name, model_save_path)
    paths_to_models = train_coles_model(monomodel=True, conf_path=conf_path, debug=debug)
    # list of (fold_i, metric_scores)
    scores = inference(mode='mono', models_info=paths_to_models, conf_path=conf_path)

    results = dict()
    for m in paths_to_models:
        results[str(m[0])] = [m[1], m[2]]
    for s in scores:
        results[str(s[0])].append(','.join([str(np.round(float(x), 5)) for x in s[1]]))
        results[str(s[0])].append(np.mean(str(s[1])))
    write_results(exp_name, results)


if __name__ == '__main__':
    main()
