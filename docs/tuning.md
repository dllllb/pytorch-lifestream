# Hyperparameters tuning

We propose a demo for hyperparameters tuning with `hydra`, `optuna` and `tensorboard`.
This is a console application located in `demo/hparam_tuning`.

# Intro

After we build a network architecture we should tune hyperparameters.
Automated tuning have benefits:

- Automated iterations over hparam set is faster than manual choice
- Automated iterations requires less operational costs
- All results logged and can be inspected together
- Iteration count limit allows measuring the quality improvement with fixed resources.
- hparam optimisation tools implement effective strategy of parameter choice

Keep in mind that is just a tool for hparam iteration.
You should decide which parameters should be tuned and define a search space.

Usually initial setup of automated hparam tuning requires a time. We hope our demo will make it easier for you.

# Used tools

We use `hydra` framework to manage app configuration, parameters and results.
`hydra` multirun mode can run the same application with multiple different configurations.
`optuna` plugin for hydra implements smart choice of next hparam set.
Results are stored in hydra output folder.

We log a results to tensorboard to show it as charts.

# Guide to reproduce

`demo/hparam_tuning/` should be working directory for this demo.

## 1. Data preprocessing

Aims of preprocessing:

- split the data into folds
- use `ptls.preprocessing.fit_transform` to convert data to `ptls` compatible format.

### Data split

In this demo we use 6 datasplits: 1 for tuning and 5 for evaluation.
We use cross-validation stratified fold splits.
This means that we use `5/6` samples for training and `1/6` samples for testing.

You can use a different setting, but there should be multiple parts to estimate the variance when testing.
Also, the validation part should be separated from the test part.

### Data preprocessing

After data split we make `ptls.preprocessing` for each part. In this demo we do it with `spark`.

We save separately the preprocessed data for each fold. This multiply required storage space, 
but it's easy to maintenance.

We include unlabeled data to train part.
You can use both data labeled and unlabeled for unsupervised training.
You can filter unlabeled data with `iterable_processing.TargetEmptyFilter` for supervised training.

After preprocessing data stored in `parquet` format.

### Scripts

- `sh bin/data-preprocessing.sh` spark-submit command for run
- `data_preprocessing.py` main program
- `conf/data_preprocessing.yaml` config for data preprocessing

## 2. Model tuning script

Python program with model train, evaluate, logging and fold_id selection based on valid of test mode.

Some hparams should be configured via config file.

Run it in `test` mode first.

- you can make sure everything works correctly
- you get mean quality for default hparams
- you get std and confidence interval for model quality

### Scripts

- `tuning.py` tuning script
- `conf/simple_config.yaml` default config
- `python tuning.py --config-name=simple_config mode=test` command for run

## 3. Tuning

Extends your config with multirun `hydra` settings, `optuna` settings and params search space.

We prepare a separate config for tuning run.

`main` in tuning script saves config in `{hydra.cwd}/conf_override/config.yaml` for future reuse during testing


### Scripts

- `tuning.py` the same tuning script
- `conf/one_layer_head.yaml` config with tuning params
- `python tuning.py --multirun --config-name=one_layer_head mode=valid` command for run

## 4. Test the best hparams (option 1)

- Check tensorboard `hparams` tab
- Choose the best run
- Check `run name` and `hydra.reuse_cmd` hparam.
- Run test with best config: `python tuning.py ---config-name=one_layer_head {hydra.reuse_cmd} mode=test`
with `{hydra.reuse_cmd}` resolve.

`{hydra.reuse_cmd}` looks like path to best config and adding overrieds to whole config: `+conf_override@=config`

This option overrides the main config `one_layer_head` by one which was used during best run. 
This option keeps `hydra` settings which are in `one_layer_head`.

## 4. Test the best hparams (option 2)

- Check tensorboard `hparams` tab
- Choose the best run
- Check `run name` and `hydra.cwd` hparam.
- Run test with best config: `python tuning.py --config-path={hydra.cwd}/.hydra --config-name=config mode=test`
with `{hydra.cwd}` resolve.

This option use `config.yaml` from `hydra` log of best run.
This option loose `hydra` settings which are in `one_layer_head` cause hydra don't log his settings in `config.yaml`.

# Result logging

Results on each fold stored to:

- tensorboard with logger selected in `tuning.model_run.trainer` with calling `log` method in your LightningModule
- hydra output folder (see `conf.hydra.sweep.dir`)

Mean results on test stores to:

- tensorboard with separate logger with name `{conf.mode}_mean`
Only hparams and final metrics are saved.
- hydra output folder

hparams with final metrics stores to:

- tb logger selected in `tuning.model_run.trainer` during validation
- logger with name `{conf.mode}_mean` during test

Log file in hydra output folder have a tensorboard run version.
And tensorboard run have `hydra.cwd` which are hydra output folder in `hparams`

For better reading `hparams` are saved as flat list without hierarchy.
Full job config are stored in `{hydra.cwd}/.hydra/config.yaml`
Copy of full job config are stored in `{hydra.cwd}/conf_override/config.yaml`
