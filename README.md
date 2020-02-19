# Setup

```sh
# download Miniconda from https://conda.io/
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# execute intall file
bash Miniconda*.sh
# relogin to apply PATH changes
exit

# install pytorch
conda install pytorch -c pytorch
# check CUDA version
python -c 'import torch; print(torch.version.cuda)'
# chech torch GPU
# See question: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
python -c 'import torch; print(torch.rand(2,3).cuda())'

# clone repo
git clone git@bitbucket.org:dllllb/dltranz.git

# install dependencies
cd dltranz
pip install -r requirements.txt

# download datasets
./get-data.sh

# convert datasets from transaction list to features for metric learning
./make-datasets.sh
```

# Run scenario

## age-pred dataset
### Main scenario, best params

```sh
cd dltrans/opends

# Train the metric learning model
python metric_learning.py --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json

# Run inference with the pretrained mertic learning model and get embeddings for each customer
python ml_inference.py --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

# Train a supervised model and save scores to the file
python -m scenario_age_pred fit_target --conf conf/age_pred_dataset.hocon conf/age_pred_target_params_train.json

# Train a special model for fine-tuning 
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model. 
python metric_learning.py --conf conf/age_pred_dataset.hocon conf/age_pred_ml_fintuning_train.json.json
# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m scenario_age_pred fit_finetuning --conf conf/age_pred_dataset.hocon conf/age_pred_finetuning_params_train.json

# Train the Contrastive Predictive Coding (CPC) model; inference
python train_cpc.py --conf conf/age_pred_dataset.hocon conf/age_pred_cpc_params_train.json
python ml_inference.py --conf conf/age_pred_dataset.hocon conf/age_pred_cpc_params_inference.json
# fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_age_pred fit_finetuning \
    params.pretrained_model_path="models/age_pred_cpc_model.p" \
    output.test.path="../data/age-pred/finetuning_cpc_scores_$SC_AMOUNT"/test \
    output.valid.path="../data/age-pred/finetuning_cpc_scores_$SC_AMOUNT"/valid \
    --conf conf/age_pred_dataset.hocon conf/age_pred_finetuning_params_train.json

# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches

# check the results
cat runs/scenario_age_pred.csv
```

### Semi-supervised setup
```sh
cd dltrans/opends

export SC_DEVICE="cuda"

# run semi supervised scenario
./scenario_age_pred/bin/scenario_semi_supervised.sh

# check the results
cat runs/semi_scenario_age_pred_*.csv

```

### Test model configurations
```sh
cd dltrans/opends

export SC_DEVICE="cuda"

# run all scenarios or select one
./scenario_age_pred/bin/*.sh

# check the results
cat runs/scenario_age_pred_*.csv

```


## gender dataset
### Main scenario, best params

```sh
cd dltrans/opends

# Train the metric learning model
python metric_learning.py --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json

# Run inference with the pretrained mertic learning model and get embeddings for each customer
python ml_inference.py --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# Train a supervised model and save scores to the file
python -m scenario_gender fit_target --conf conf/gender_dataset.hocon conf/gender_target_params_train.json

# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m scenario_gender fit_finetuning --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

# Train the Contrastive Predictive Coding (CPC) model; inference 
python train_cpc.py --conf conf/gender_dataset.hocon conf/gender_cpc_params_train.json
python ml_inference.py --conf conf/gender_dataset.hocon conf/gender_cpc_params_inference.json
# fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning \
    params.pretrained_model_path="models/gender_cpc_model.p" \
    output.test.path="../data/gender/finetuning_cpc_scores"/test \
    output.valid.path="../data/gender/finetuning_cpc_scores"/valid \
    --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

# TODO: remove it after end of experiment
# Train the metric learning model with transformer
python metric_learning.py params.device="$SC_DEVICE" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
# python ml_inference.py params.device="$SC_DEVICE" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json

python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_25.pth" output.path="../data/gender/embeddings_transf_e0025" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_50.pth" output.path="../data/gender/embeddings_transf_e0050" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_75.pth" output.path="../data/gender/embeddings_transf_e0075" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_100.pth" output.path="../data/gender/embeddings_transf_e0100" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_125.pth" output.path="../data/gender/embeddings_transf_e0125" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_150.pth" output.path="../data/gender/embeddings_transf_e0150" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_175.pth" output.path="../data/gender/embeddings_transf_e0175" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json
python ml_inference.py params.device="$SC_DEVICE" model_path.model="models/gender_checkpoints/transf_model_200.pth" output.path="../data/gender/embeddings_transf_e0200" --conf conf/gender_dataset.hocon conf/gender_transformer_params.json

python -m scenario_gender compare_approaches --n_workers 1 --skip_baseline --skip_linear --skip_xgboost --target_score_file_names \
    --ml_embedding_file_names \
    "embeddings_transf_e0200.pickle" \
    "embeddings_transf_e0175.pickle" \
    "embeddings_transf_e0150.pickle" \
    "embeddings_transf_e0125.pickle" \
    "embeddings_transf_e0100.pickle" \
    "embeddings_transf_e0075.pickle" \
    "embeddings_transf_e0050.pickle" \
    "embeddings_transf_e0025.pickle"


python -m scenario_gender compare_approaches --n_workers 1 --skip_baseline --skip_linear --skip_xgboost --target_score_file_names \
    --ml_embedding_file_names \
    "embeddings_transf_e0125.pickle"
    

# with starter to embedding
# with lr sheduler
# No pos encoding
# No input mask
# with base network
name
lgb_embeds: embeddings_transf_e0025.pickle           0.8548  0.8382  0.8715 0.0120  [0.837 0.852 0.854 0.864 0.867]            0.8513  0.8468  0.8559 0.0033  [0.848 0.848 0.852 0.853 0.856]
lgb_embeds: embeddings_transf_e0050.pickle           0.8609  0.8422  0.8796 0.0135  [0.841 0.855 0.864 0.870 0.875]            0.8557  0.8524  0.8589 0.0023  [0.852 0.856 0.856 0.857 0.858]
lgb_embeds: embeddings_transf_e0075.pickle           0.8590  0.8405  0.8776 0.0134  [0.839 0.854 0.861 0.869 0.872]            0.8582  0.8547  0.8616 0.0025  [0.855 0.856 0.859 0.860 0.861]
lgb_embeds: embeddings_transf_e0100.pickle           0.8587  0.8385  0.8789 0.0146  [0.836 0.853 0.862 0.870 0.873]            0.8512  0.8463  0.8561 0.0035  [0.847 0.849 0.850 0.855 0.855]
lgb_embeds: embeddings_transf_e0125.pickle           0.8609  0.8417  0.8801 0.0138  [0.841 0.852 0.869 0.871 0.872]            0.8578  0.8547  0.8610 0.0022  [0.855 0.857 0.858 0.858 0.861]
lgb_embeds: embeddings_transf_e0150.pickle           0.8583  0.8363  0.8803 0.0158  [0.832 0.854 0.864 0.870 0.871]            0.8511  0.8470  0.8552 0.0029  [0.847 0.851 0.851 0.852 0.855]
lgb_embeds: embeddings_transf_e0175.pickle           0.8581  0.8364  0.8798 0.0156  [0.834 0.853 0.860 0.869 0.874]            0.8595  0.8554  0.8635 0.0029  [0.855 0.859 0.859 0.862 0.862]



# with starter to embedding
# without lr sheduler
# No pos encoding
# No input mask
# with base network
name
lgb_embeds: embeddings_transf_e0025.pickle           0.8521  0.8354  0.8689 0.0121  [0.832 0.852 0.855 0.861 0.861]            0.8405  0.8364  0.8446 0.0029  [0.838 0.839 0.839 0.842 0.845]
lgb_embeds: embeddings_transf_e0050.pickle           0.8640  0.8460  0.8820 0.0130  [0.844 0.861 0.865 0.873 0.877]            0.8565  0.8529  0.8601 0.0026  [0.853 0.856 0.857 0.857 0.860]
lgb_embeds: embeddings_transf_e0075.pickle           0.8583  0.8388  0.8778 0.0141  [0.839 0.851 0.860 0.866 0.876]            0.8555  0.8504  0.8607 0.0037  [0.852 0.852 0.855 0.857 0.861]
lgb_embeds: embeddings_transf_e0100.pickle           0.8527  0.8338  0.8717 0.0137  [0.830 0.852 0.857 0.857 0.867]            0.8532  0.8494  0.8570 0.0027  [0.850 0.851 0.854 0.855 0.856]
lgb_embeds: embeddings_transf_e0125.pickle           0.8560  0.8338  0.8781 0.0160  [0.830 0.851 0.861 0.867 0.870]            0.8566  0.8549  0.8583 0.0012  [0.855 0.856 0.857 0.858 0.858]


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_gender compare_approaches

# check the results
cat runs/scenario_gender.csv
```

### Semi-supervised setup
```sh
cd dltrans/opends

export SC_DEVICE="cuda"

# run semi supervised scenario
./scenario_gender/bin/scenario_semi_supervised.sh

# check the results
cat runs/semi_scenario_gender_*.csv

```

### Test model configurations

```sh
cd dltrans/opends

export SC_DEVICE="cuda"

# run all scenarios or select one
./scenario_gender/bin/*.sh

# check the results
cat runs/scenario_gender_*.csv
```

## tinkoff dataset: Story recommendation with cold start

```sh
cd dltrans/opends

# Train metric learning model
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json

# With pretrained mertic learning model run inference ang take embeddings for each customer
python ml_inference.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json

# Run estimation for different approaches
# Check some options with `--help` argument
dltrans/opends $ 
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

## tinkoff dataset: Socio-demographic characteristics by transactions
Note. This is the same dataset as in previous step.
You don't need retun metric learning model train and recalculate embeddings.
Use the ones you have already prepared.

```sh
cd dltrans/opends

# (If wasn't ran before) Train the metric learning model
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json

# (If wasn't ran before) # Run inference with the pretrained mertic learning model and take embeddings for each customer
python ml_inference.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_tin_cls compare_approaches

# check the results
cat runs/scenario_tin_cls_*.csv

```


## common scenario (work in progress)

### Train metric learning model

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model params in json file

```sh
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json
```

### With pretrained mertic learning model run inference ang take embeddings for each customer

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model inference params in json file

```sh
python ml_inference.py --conf conf/dataset.hocon conf/ml_params_inference.json
```

### Use embeddings for each customer as a features for specific machine learning problem

See code in notebooks in `notebooks/`.
