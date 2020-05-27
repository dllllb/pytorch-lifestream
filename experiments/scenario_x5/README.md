# Get data

```sh
cd experiments/scenario_x5

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_ч5
export SC_DEVICE="cuda"

# Train a supervised model and save scores to the file
python -m scenario_x5 fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_x5 compare_approaches --n_workers 1 \
   --score_file_names "target_scores"


# check the results
cat results/scenario.csv

```
