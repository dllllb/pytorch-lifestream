# Get data

```sh
cd scenario_alpha_battle

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make-datasets-spark.sh
```

# Main scenario, baselines

```sh
cd scenario_alpha_battle
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```

# Main scenario, unsupervised methods

```sh
cd scenario_alpha_battle
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

```

This is a big dataset. Only unsupervised task are presented.
Run `python -m ptls.pl_train_module --config-dir conf --config-name params` with specific `params.yaml` file and use
scripts from `bin/embeddings_by_epochs` to get score on downstream task by epochs.
See `figures.ipynb` for visualisation.


# Comments on experiments 

`results/scenario_alpha_battle_mlmnsp_tabformer_gpt.txt`
- gpt_[hidden_size]_[pooling] - GPT2 model learned like language model on transactions, which autoregressivly predicts classes of each feature of next transaction. Pooling strategies to get a whole sequence embedding are: **out** - last transaction embedding, **outstat** - global min|max|mean|std statistics concatenation over model output, **trxstat** - global min|max|mean|std statistics concatenation over trx_encoder output, **trxstat_out** - **trxstat** union with **out**.
- gptrnn_[hidden_size]\_[pooling] - the same as *gpt_[hidden_size]_[pooling]*, but was used RNN model instead of transformer.
- mlm_[hidden_size]_[pooling] - joined pretrain task Masked Language Model + Next Sequence Prediction, Longformer model. Pooling strategies to get a whole sequence embedding are: **cls** - get only [CLS] token as sequence embedding, **stat** - **cls** union with global min|max|mean|std statistics concatenation over trx_encoder output.
- tab_[hidden_size]\_[pooling] - pretrain approach like in Tabformer, where we are predicting class of random feature of transaction intead of the whole transaction embedding as in MLM, Longformer model. Pooling are the same as in *mlm_[hidden_size]_[pooling]*.

`results/scenario_mles_rnn_transf.txt`
- rnn[hs] - classic MLES pretrain with GRU model under the hood, sequence embedding is last hidden_state of RNN.
- rnn*_shuffle - all transactions are randomly shuffled for every time series over the time dimension.
- rnn*_stat - sequence embedding is global min|max|mean|std statistics concatenation over the trx_encoder output + last hidden_state of RNN.
- rnn*_wotimefeat - checked hypotesys about negative impact of time features (like month of year, etc) on MLES learning.
- tabformer* - tabformer pretrain task with Longformer model.
- transf* - MLES pretrain with Longformer model.