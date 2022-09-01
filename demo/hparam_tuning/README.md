# Hyperparameter tuning demo with hydra+optuna+tensorboard

## Fast run

Run the following commands:
```shell
# load data
sh bin/get-data.sh

# split folds and preprocessing the data
sh bin/data-preprocessing.sh

# hyperparameter optimisation with valid folds
# with total steps limitation to make it faster
python train_model.py --multirun --config-name=one_layer experiment_name=fast_test \
  trainer.max_epochs=3 trainer.limit_train_batches=50 hydra.sweeper.n_trials=10

# track logs in multirun/ folder
# track quality with tensorboard
tensorboard --logdir lightning_logs/ 

# Find the best run in tensorboard HPARAMS tab
# get hydra.cwd from this run and concat `.hydra` folder to get path_to_best_config
# {path_to_best_config} will be like demo/hparam_tuning/multirun/fast_test/2022-08-31/10-12-14/1/.hydra

# Estimate a final quality of best hparam set on test folds
# config.yaml in {path_to_best_config} overridden with the same total steps limitation at was before
python train_model.py --config-path={path_to_best_config} --config-name=config mode=test

```

## Full tuning for comparing two approaches

In difference of previous run we will compare two models:
- with one layer head
- with two layers head

We run tuning for first and take the best params.
Next we run tuning for second and take the best params.
Finally, we compare the first and second approaches with his best params.

## Work dir preparation

It's better to clear working directory before new experiment

```shell
rm -r multirun/
rm -r output/
rm -r lightning_logs/
```

We expect that data is already loaded and split with scripts:
```shell
# load data
sh bin/get-data.sh

# split folds and preprocessing the data
sh bin/data-preprocessing.sh
```

## Run 1st approach (with fixed max_epochs)
```shell
python train_model.py --multirun --config-name=one_layer

python train_model.py --config-path={path_to_best_one_layer_config} --config-name=config mode=test

```

## Run 2nd approach (with early stopping)
```shell
python train_model.py --multirun  --config-name=two_layers

python train_model.py --config-path={path_to_best_two_layers_config} --config-name=config mode=test

```
