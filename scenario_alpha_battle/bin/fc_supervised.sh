ckpt_path=composition_results
max_batch=10000
echo "Training phase is up"
for type_ in raw composite_1 composite_2
do
  echo "Started to run $type_ supervised finetuning"
  rm -rf $ckpt_path/$type_/checkpoints
  mkdir -p $ckpt_path/$type_/checkpoints 
  python -m fc_train \
  pl_module.seq_encoder.hidden_size=1024 \
  +pl_module.loss='{_target_: ptls.frames.coles.losses.MarginLoss, margin: 0.2, beta: 0.4}' \
  +pl_module.loss.pair_selector='{_target_: ptls.frames.coles.sampling_strategies.HardNegativePairSelector, neg_count: 5}' \
  ~pl_module.loss.sampling_strategy \
  +pretrained="models/mles_model.p" \
  +type_=test\
  +limit_train_batches=$max_batch\
  --config-dir conf --config-name fc_mles_params \
  >> composition_results/$type_/sdout.txt

  echo "Finetuning ended"

  python -m fc_eval_sup \
  +ckpts=compression_results/$type_/checkpoints\
  +pretrained="models/mles_sup_model.p" \
  +type_=$type_\
  +limit_train_batches=$max_batch\
  +limit_valid_batches=$max_batch\
  +need_pretrain=false\
   --config-dir conf --config-name fc_fit_finetuning_mles

done
echo "Done"


  
