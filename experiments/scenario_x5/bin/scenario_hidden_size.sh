for SC_HIDDEN_SIZE in 0064 0160 0480 0800
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path="models/x5_mlm__$SC_SUFFIX.p" \
      --conf conf/mles_params.hocon
  python ../../pl_inference.py \
      model_path="models/mles__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf conf/mles_params.hocon
done

for SC_HIDDEN_SIZE in 1600
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      params.train.batch_size=128 \
      model_path="models/x5_mlm__$SC_SUFFIX.p" \
      --conf conf/mles_params.hocon
  python ../../pl_inference.py \
      model_path="models/mles__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf conf/mles_params.hocon
done

# Compare
rm results/scenario_x5__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_x5__hidden_size.txt",
      auto_features: ["../data/emb__hidden_size_*.pickle"]'

