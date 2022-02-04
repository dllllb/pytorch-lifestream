# Get data

```sh
cd experiments/scenario_alpha_battle

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make-datasets-spark.sh
```

# Main scenario, baselines

```sh
cd experiments/scenario_alpha_battle
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```

# Main scenario, unsupervised methods

```sh
cd experiments/scenario_alpha_battle
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

```

This is a big dataset. Only unsupervised task are presented.
Run `python ../../pl_train_module.py --conf conf/params.hocon` with specific `params.hocon` file and use
scripts from `bin/embeddings_by_epochs` to get score on downstream task by epochs.
See `figures.ipynb` for visualisation.

