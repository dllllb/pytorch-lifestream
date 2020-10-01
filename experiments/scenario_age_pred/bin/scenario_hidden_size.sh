for SC_HIDDEN_SIZE in 1600 1200 0800 0480 0224 0160 0096 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python ../../metric_learning.py \
      params.device="$SC_DEVICE" \
      params.train.batch_size=64 \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

for SC_HIDDEN_SIZE in 0800 0480 0224 0160 0096 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python ../../ml_inference.py \
      params.device="$SC_DEVICE" \
      model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
      output.path="data/emb__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

for SC_HIDDEN_SIZE in 2400 1600 1200
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python ../../ml_inference.py \
      params.device="$SC_DEVICE" \
      params.valid.batch_size=256 \
      model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
      output.path="data/emb__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
rm results/scenario_age_pred__hidden_size.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_age_pred__hidden_size.txt",
      auto_features: ["../data/emb__hidden_size_*.pickle"]'
