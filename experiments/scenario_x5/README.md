# Get data

```sh
cd experiments/scenario_x5

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_x5
export SC_DEVICE="cuda"

sh bin/run_all_scenarios.sh

```
