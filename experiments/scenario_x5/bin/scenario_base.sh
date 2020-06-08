# Base model
export SC_EPOCH_N=1

export SC_SUFFIX="base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.rnn.type="gru" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_base.csv" \
    --n_workers 1 --models lgb --embedding_file_names \
    "emb__base.pickle"
