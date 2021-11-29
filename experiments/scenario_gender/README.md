# Get data

```sh
cd experiments/scenario_gender

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_gender
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number


sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```
