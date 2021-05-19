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
export SC_GROUP="lambd"
for SC_PARAMETER in 0.001 0.01 0.1
do
  export SC_SUFFIX="bt_${SC_GROUP}_${SC_PARAMETER}"
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

