export SC_SUFFIX="hidden_size_0160"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=160 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0080"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=80 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0040"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=40 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0020"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=20 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_bowl2019 compare_approaches --output_file "results/scenario_bowl2019__hidden_size.csv" \
    --embedding_file_names \
    "emb__hidden_size_0160.pickle" \
    "emb__hidden_size_0080.pickle" \
    "emb__hidden_size_0040.pickle" \
    "emb__hidden_size_0020.pickle"
            
