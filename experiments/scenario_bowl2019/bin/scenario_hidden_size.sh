export SC_SUFFIX="hidden_size_bs_0064_hs_1600"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=1600 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_1200"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=1200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0800"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=800 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0480"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=480 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0224"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=224 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0160"
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

export SC_SUFFIX="hidden_size_bs_0064_hs_0096"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=96 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=64 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_bs_0064_hs_0032"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=32 \
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
    "emb__hidden_size_bs_0064_hs_1600.pickle" \
    "emb__hidden_size_bs_0064_hs_1200.pickle" \
    "emb__hidden_size_bs_0064_hs_0800.pickle"  \
    "emb__hidden_size_bs_0064_hs_0480.pickle"  \
    "emb__hidden_size_bs_0064_hs_0224.pickle"  \
    "emb__hidden_size_bs_0064_hs_0160.pickle"  \
    "emb__hidden_size_bs_0064_hs_0096.pickle"  \
    "emb__hidden_size_bs_0064_hs_0064.pickle"  \
    "emb__hidden_size_bs_0064_hs_0032.pickle"
