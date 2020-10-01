# HiddenSize for batch_size=128
for SC_HIDDEN_SIZE in 3072 2048 1024 0512 0256 0128 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0128_hs_${SC_HIDDEN_SIZE}"
  python ../../metric_learning.py \
      params.device="$SC_DEVICE" \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
python -m scenario_gender compare_approaches --output_file "results/scenario_gender__hidden_size.csv" \
    --models lgb \
    --embedding_file_names "emb__hidden_size_bs_0128_hs_*.pickle"

# HiddenSize for batch_size=128
for SC_HIDDEN_SIZE in 1024 0512 0256 0128 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0128_hs_${SC_HIDDEN_SIZE}"
  python ../../ml_inference.py \
      params.device="$SC_DEVICE" \
      model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
      output.path="data/emb__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

for SC_HIDDEN_SIZE in 3072 2048
do
    export SC_SUFFIX="hidden_size_bs_0128_hs_${SC_HIDDEN_SIZE}"
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        params.valid.batch_size=128 \
        model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
        output.path="data/emb__$SC_SUFFIX" \
        --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
python -m scenario_gender compare_approaches --output_file "results/scenario_gender__hidden_size.csv" \
    --models lgb \
    --embedding_file_names "emb__hidden_size_bs_0128_hs_*.pickle"
