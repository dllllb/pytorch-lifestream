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
python tuning.py --multirun --config-name=one_layer_head mode=valid

# track logs in multirun/ folder
# track quality with tensorboard
tensorboard --logdir lightning_logs/ 

# Find the best run in tensorboard HPARAMS tab
# get `hydra.reuse_cmd` from this run

# Estimate a final quality of best hparam set on test folds
# config.yaml in {path_to_best_config} overridden with the same total steps limitation at was before
python tuning.py ---config-name=one_layer_head {hydra.reuse_cmd} mode=test

```


See `docs/tuning.md` for more information.
