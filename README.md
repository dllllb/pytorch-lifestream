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
```

# Run scenario

1. Convert datasets from transaction list to features for metric learning

```sh
python make_datasets.py \
    --data_path ../data/age-pred/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --output_path "../data/age-pred/all_trx.p"

python make_datasets.py \
    --data_path ../data/tinkoff/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "transaction_month" "transaction_day" \
    --cols_category "transaction_month" "transaction_day" "merchant_id" "merchant_mcc" \
    --cols_log_norm "transaction_amt" \
    --output_path "../data/tinkoff/all_trx.p"

python make_datasets.py \
    --data_path data/gender/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "tr_datetime" \
    --cols_category "mcc_code" "tr_type" "term_id" \
    --cols_log_norm "amount" \
    --output_path "../data/gender/all_trx.p"
```

2. Train metric learning model

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model params in json file

```sh
python metric_learning.py --conf conf/sber_ml_dataset.hocon conf/sber_ml_params_train.json

python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json
```

3. With pretrained mertic learning model run inference ang take embeddings for each customer

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model inference params in json file

```
python ml_inference.py --conf conf/dataset.hocon conf/ml_params_inference.json
```

4. Use embeddings for each customer as a features for specific machine learning problem

See code in notebooks in `notebooks/`.
