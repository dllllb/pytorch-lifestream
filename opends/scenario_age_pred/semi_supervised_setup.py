import subprocess

for labeled_amount in [337]: #, 675, 1350, 2700, 5400, 10800]:
    # fit target
    bash_command = [
        'python -m scenario_age_pred fit_target',
        'params.device="cuda:0"',
        f'params.labeled_amount={labeled_amount}',
        f'output.test.path="../data/age-pred/target_scores_{labeled_amount}"/test',
        f'output.valid.path="../data/age-pred/target_scores_{labeled_amount}"/valid',
        '--conf conf/age_pred_target_dataset.hocon conf/age_pred_target_params_train.json'
    ]
    bash_command = ' '.join(bash_command)

    # p = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = p.communicate()

    # fine tunning
    bash_command = [
        'python -m scenario_age_pred fit_finetuning',
        'params.device="cuda:0"',
        f'params.labeled_amount={labeled_amount}',
        f'output.test.path="../data/age-pred/finetuning_scores_{labeled_amount}"/test',
        f'output.valid.path="../data/age-pred/finetuning_scores_{labeled_amount}"/valid',
        '--conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json'
    ]
    bash_command = ' '.join(bash_command)

    # p = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = p.communicate()

    # compare approaches
    bash_command = [
        'python -m scenario_age_pred compare_approaches',
        f"--target_score_file_names target_scores_{labeled_amount} finetuning_scores_{labeled_amount}",
        f'--labeled_amount {labeled_amount}',
        f"--output_file runs/semi_scenario_age_pred_{labeled_amount}.csv"
    ]
    bash_command = ' '.join(bash_command)

    p = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = p.communicate()