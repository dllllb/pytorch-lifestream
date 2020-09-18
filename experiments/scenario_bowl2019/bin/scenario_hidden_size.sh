export SC_SUFFIX="hidden_size_0032"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=32 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=64 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


export SC_SUFFIX="hidden_size_0100"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=100 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0200"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_bowl2019 compare_approaches --output_file "results/scenario_bowl2019__hidden_size.csv" \
    --models 'lgb' --embedding_file_names \
    "emb__hidden_size_0032.pickle" \
    "emb__hidden_size_0064.pickle" \
    "emb__hidden_size_0100.pickle" \
    "emb__hidden_size_0200.pickle"

