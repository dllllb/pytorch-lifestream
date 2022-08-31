# Hyperparameter tuning demo with hydra+optuna+tensorboard

## Intro


## Fast run

Run the following commands:
```shell
# load data
sh bin/get-data.sh

# split folds and preprocessing the data
sh bin/data-preprocessing.sh

# hyperparameter optimisation with valid folds
python train_model.py --multirun

# track logs in multirun/ folder
# track quality with tensorboard
tensorboard --logdir lightning_logs/ 

# Find the best run in tensorboard HPARAMS tab
# gen hydra.cwd from this run
# {path_to_best_config} will be like demo/hparam_tuning/multirun/model_1/2022-08-31/10-12-14/1/.hydra

# Estimate a final quality of best hparam set on test folds
python train_model.py --config-path={path_to_best_config} mode=test

```
