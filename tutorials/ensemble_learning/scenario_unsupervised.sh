# Unsupervised learning scenario for parameters specified below

# Some architectural setup
model=mles
pretrained_encoder_path=""
setup="composite_1"

# Additional settings
max_batch_train=1000000
max_batch_val=1000000
need_fedot_pretrain=false
ckpt_dir=checkpoints
embeddings_dir=embeddings

train_log=$ckpt_dir/"$(echo $model)"_train_log.txt
mkdir $ckpt_dir
touch $train_log

# Training & compression
echo "Unsupervised compression of $model in $setup mode started"
python -m ptls.fedcore_compression.fc_train \
    +pretrained=$pretrained_encoder_path\
    +save_encoder=models/"$(echo $model)"_for_finetuning.p \
    +need_evo_opt=false\
    +setup=$setup \
    --config-dir conf --config-name "$(echo $model)"_params \
    >> $train_log

# Embedding generation and computational efficacy estimation 
# (to turn it on the n_batches_computational > 0 should be specified)
echo "Inference stage of $model in $setup mode started"
mkdir $embeddings_dir
python -m ptls.fedcore_compression.fc_inference --config-dir conf --config-name "$(echo $model)"_params \
  +inference.output=$embeddings_dir \
  +inference_model=$ckpt_dir \
  +limit_predict_batches=$max_batch_train \
  +n_batches_computational=${n_batches_computational:- 0}
