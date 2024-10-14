ckpt_path=composition_results
batch_limit=10000
echo "Training phase is up"
for type_ in raw composite_1 composite_2
do
  echo "Started to run $type_ scenario"
  rm -rf $ckpt_path/$type_/checkpoints
  mkdir -p $ckpt_path/$type_/checkpoints 
  python -m fc_train --config-dir conf --config-name fc_mles_params \
  +type_="${type_}"\
  +need_pretrain=false\
  +limit_train_batches=$batch_limit\
  +limit_valid_batches=$batch_limit\
    >> composition_results/$type_/sdout.txt
  
  echo "Embeddings generation"
  rm -rf $ckpt_path/$type_/scores
  mkdir -p $ckpt_path/$type_/scores 
  python -m fc_inference --config-dir conf --config-name fc_mles_params\
    +type_="${type_}"\
    ++inference.output="composition_results/$type_/scores"\
    +ckpts="$ckpt_path/$type_/checkpoints"\
    +n_batches_computational=2\
    +max_batch=16
  echo "Embeddings for $type_ generated"
  
  python -m fc_emb_eval --config-dir conf --config-name fc_embeddings_validation_short \
    +type_="${type_}" +emb_path="composition_results"
done
echo "Training phase is done"
