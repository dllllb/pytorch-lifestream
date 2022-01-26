#for SC_HIDDEN_SIZE in 1024 0512 0256 0128 0064
for SC_HIDDEN_SIZE in 2048
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      data_module.train.batch_size=128 \
      data_module.valid.batch_size=128 \
      model_path="models/mlm__$SC_SUFFIX.p" \
      --conf conf/mles_params.hocon
  python ../../pl_inference.py \
      model_path="models/mlm__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf conf/mles_params.hocon
done

for SC_HIDDEN_SIZE in 0128 0064
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path="models/mlm__$SC_SUFFIX.p" \
      --conf conf/mles_params.hocon
  python ../../pl_inference.py \
      model_path="models/mlm__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf conf/mles_params.hocon
done

# Compare
rm results/scenario_alpha_battle__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_alpha_battle__hidden_size.txt",
      auto_features: ["../data/emb_mles__hidden_size_*.pickle"]'

