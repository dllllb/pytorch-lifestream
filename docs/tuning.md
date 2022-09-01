# Hyperparameters tuning

We propose a demo for hyperparameters tuning with `hydra`, `optuna` and `tensorboard`.
This is console application located in `demo/hparam_tuning`.

## Intro

After we build a network architecture we should tune hyperparameters.
Automated tuning have a benefits:

- Automated iterations over hparam set is faster than manual choice
- Automated iterations requires less operational costs
- All results logged and can be inspected together
- Iteration count limit allow measure quality improvement with fixed resources.
- hparam optimisation tools implement effective strategy of parameter choice

Keep in mind that is just a tool for hparam iteration.
You should decide which parameters should be tuned and define a search space.

Usually initial setup of automated hparam tuning requires a time. We hope our demo will make it easier for you.

## Used tools

We use `hydra` framework to manage app configuration, parameters and results.
`hydra` multirun mode can run the same application with multiple different configurations.
`optuna` plugin for hydra implements smart choice of next hparam set, better than random search.
Results are stored in hydra output folder.

We log a results to tensorboard to show it as charts.

## Parts of demo

`demo/hparam_tuning/` should be working directory for this demo.

### Data

This demo is with small dataset. This means than model quality on different subsets may have a big variance.
This may be a cause of choice a configuration which randomly shows improvements.
We use mean qualify on 3 folds to measure model result when we check different hparams.
This is validation set with 3 folds.
Next we chose a hparams which the best on validation set.
Next we test this hparams on 5 test folds.

We have `3 + 5 = 8` folds in total.

You can choose another folds count but you should keep valid and test parts.
It's better to have more than 1 folds to measure quality variance.

We split a data into `8` folds with crossvalidation.
Next we make preprocessing for them save train and test parts for each fold.
All of this made with `bin/data-preprocessing.sh` script.

### Train script and config

`train_model.py` is a main script of demo. 
`conf/one_layer.yaml` and `conf/two_layers.yaml` are configs.

### `train_model.get_fold_list`

Returns a list of folds number depending on `conf.mode`:

- `[0, 1, 2]` for valid
- `[3, 4, 5, 6, 7]` for test

### `train_model.main`

- Run model train and estimate for each fold
- Log the results
- Returns `results_mean` for `optuna` plugin

### `train_model.model_run`

Train and estimate a model on one fold.

Datamodule and LightningModule are created based on conf with `get_data_module` and `get_pl_module`.

The model with train framework should be here. Pretrain (if required) also should be here.

### Result logging

Results on each fold stored to:

- tensorboard with logger selected in `train_model.model_run.trainer` with calling `log` method in your LightningModule
- hydra output folder (see `conf.hydra.sweep.dir`)

Mean results stores to:

- tensorboard with separate logger with name `{conf.mode}_mean`
Only hparams and final metrics are saved.
- hydra output folder

Log in hydra output folder have a tensorboard run version.
And tensorboard run have `hydra.cwd` which are hydra output folder in `hparams`

For better reading `hparams` are saved as flat list without hierarchy.
Full config are stored in `{hydra.cwd}/.hydra/config.yaml`

