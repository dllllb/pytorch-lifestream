# Get data

```sh
cd experiments/scenario_tinkoff

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets.sh
```

## tinkoff dataset: Story recommendation with cold start

```sh
cd experiments/scenario_tinkoff
export SC_DEVICE="cuda"

# Train metric learning model
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json

# With pretrained mertic learning model run inference ang take embeddings for each customer
python ml_inference.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json

# Run estimation for different approaches
# Check some options with `--help` argument
rm runs/scenario_tinkoff.json

python -m scenario_tinkoff train --name 'baseline_const' --user_layers 1 --item_layers 1 --max_epoch 2

python -m scenario_tinkoff train --name 'no user features' --user_layers 1 --item_layers E
python -m scenario_tinkoff train --name 'ml embeddings'  --use_embeddings --user_layers 1T --item_layers E1
python -m scenario_tinkoff train --name 'transactional stat'  --use_trans_common_features --use_trans_mcc_features --user_layers 1T --item_layers E1
python -m scenario_tinkoff train --name 'social demograpy' --use_gender --user_layers 1T --item_layers E1

# check the results
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"

cat "runs/scenario_tinkoff.csv"
```
