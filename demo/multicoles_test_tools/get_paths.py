import os
import pathlib


def create_experiment_folder(experiment_name):
    folder_name = './' + str(experiment_name)
    models = os.path.join(folder_name, 'models')
    logs = os.path.join(folder_name, 'logs')
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return folder_name, models, logs


def write_results(exp_name, results):
    files = set([x for x in os.listdir('./') if x.endswith('.csv')])
    exp_file, i = exp_name + '.csv', 0
    while True:
        i += 1
        if exp_file in files:
            exp_file = exp_name + '_' + str(i) + '.csv'
        else:
            break
    with open(exp_file, 'w') as f:
        headers = ';'.join(['AllScores', 'AverageScores', 'Fold', 'TBName', 'PathToModel'])+'\n'
        f.write(headers)
        for k, v in results.items():
            line = ';'.join([v[-2], v[-1], k, v[0], v[1]])+'\n'
            f.write(line)
