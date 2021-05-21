# Check COLEs with split_count=2
python ../../pl_train_module.py \
    data_module.train.split_strategy.split_count=2 \
    data_module.valid.split_strategy.split_count=2 \
    params.validation_metric_params.K=1 \
    model_path="models/mles_model2.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py    \
    model_path="models/mles_model2.p" \
    output.path="data/emb__bt_mles2" \
    --conf conf/mles_params.hocon
rm results/res_bt_mles.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_mles.txt",
      auto_features: ["../data/emb__bt_mles*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_mles.txt


# Lambda in loss
for SC_PARAMETER in 0.06  # 0.001 0.01 0.1 0.02 0.04 0.005 0.002
do
  export SC_SUFFIX="bt_lambd_${SC_PARAMETER}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.train.lambd=${SC_PARAMETER} \
      model_path="models/gender_mlm__$SC_SUFFIX.p" \
      --conf "conf/barlow_twins_params.hocon"
  python ../../pl_inference.py \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
      --conf "conf/barlow_twins_params.hocon"
done
# Compare
rm results/res_bt_lambd.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_lambd.txt",
      auto_features: ["../data/emb__bt_lambd_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_lambd.txt


# hidden_size
for SC_PARAMETER in 1536 # 128 256 512 768
do
  export SC_SUFFIX="bt_hs_${SC_PARAMETER}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_PARAMETER} \
      model_path="models/gender_mlm__$SC_SUFFIX.p" \
      --conf "conf/barlow_twins_params.hocon"
  python ../../pl_inference.py \
    inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
      --conf "conf/barlow_twins_params.hocon"
done
# Compare
rm results/res_bt_hs.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_hs.txt",
      auto_features: ["../data/emb__bt_hs_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_hs.txt

# prj
for SC_PARAMETER in 256  # 64 128 256 512 768
do
  export RNN_SIZE=2048
  export SC_SUFFIX="bt_prj_${RNN_SIZE}_${SC_PARAMETER}"
  export PRJ_SIZE=${SC_PARAMETER}
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size="${RNN_SIZE}" \
      "params.head_layers=[[Linear, {in_features: ${RNN_SIZE}, out_features: ${PRJ_SIZE}, bias: false}], [BatchNorm1d, {num_features: ${PRJ_SIZE}}], [ReLU, {}], [Linear, {in_features: ${PRJ_SIZE}, out_features: ${PRJ_SIZE}, bias: false}], [BatchNorm1d, {num_features: ${PRJ_SIZE}, affine: False}]]" \
      model_path="models/gender_mlm__$SC_SUFFIX.p" \
      --conf "conf/barlow_twins_params.hocon"
  python ../../pl_inference.py \
    inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
      --conf "conf/barlow_twins_params.hocon"
done
# Compare
rm results/res_bt_prj.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_prj.txt",
      auto_features: ["../data/emb__bt_prj_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_prj.txt


# batch_size
for SC_PARAMETER in 64 # 256
do
  export SC_SUFFIX="bt_bs_${SC_PARAMETER}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      data_module.train.batch_size=${SC_PARAMETER} \
      model_path="models/gender_mlm__$SC_SUFFIX.p" \
      --conf "conf/barlow_twins_params.hocon"
  python ../../pl_inference.py \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
      --conf "conf/barlow_twins_params.hocon"
done
# Compare
rm results/res_bt_bs.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_bs.txt",
      auto_features: ["../data/emb__bt_bs_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_bs.txt



export SC_SUFFIX="bt_tuning_new"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py         inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
