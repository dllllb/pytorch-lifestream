import os
import shutil


def create_experiment_folder(experiment_name):
    folder_name = './' + str(experiment_name)
    models = os.path.join(folder_name, 'models')
    logs = os.path.join(folder_name, 'logs')
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return folder_name, models, logs


def write_results(exp_name, result):
    files = set([x for x in os.listdir(f'./{exp_name}/') if x.endswith('.csv')])
    exp_file = f'./{exp_name}/' + exp_name + '.csv'
    add_headers = False if exp_name + '.csv' in files else True
    with open(exp_file, 'a') as f:
        if add_headers:
            headers = ';'.join(['AllScores', 'AverageScores', 'Fold', 'NetType', 'TBName', 'PathToModel'])+'\n'
            f.write(headers)
        line = ';'.join([result['scores'], result['mean_score'], result['fold_i'], result['net_type'],
                         result['tb_name'], result['model_path']])+'\n'
        f.write(line)


def save_config_copy(conf_path, exp_name):
    conf_name = os.path.basename(conf_path)
    shutil.copyfile(conf_path, os.path.join(exp_name, conf_name))


def add_next_ind(fold, name):
    files = set([x for x in os.listdir(fold)])
    prop_name = name
    i = -1
    while True:
        if prop_name in files:
            i += 1
            prop_name = '_'.join([name, str(i)])
        else:
            break
    return prop_name
