python ../../pl_fit_target.py \
  data_module.train.drop_last=true \
  logger_name="bf_ftning_v01" \
  params.pretrained.lr=0.001 \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon


