# HiddenSize
# for SC_HIDDEN_SIZE in 3072 2048 1024 0512 0256 0128 0064 0032
for SC_HIDDEN_SIZE in 3072 2048 1024 0512 0256 0128 0064 0032
do
  export SC_SUFFIX="hidden_size_${SC_HIDDEN_SIZE}"
  python ../../metric_learning.py \
      params.device="$SC_DEVICE" \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path.model="models/mles__$SC_SUFFIX.p" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
  python ../../ml_inference_lazy.py \
      params.device="$SC_DEVICE" \
      model_path.model="models/mles__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
rm results/scenario_rosbank__hidden_size.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank__hidden_size.txt",
      auto_features: ["../data/emb_mles__hidden_size_*.pickle"]'
# less -S results/scenario_rosbank__hidden_size.txt
