# HiddenSize for batch_size=128
export SC_SUFFIX="hidden_size_bs_0128_hs_3072"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=3072 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_2048"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=2048 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_1024"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=1024 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0512"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=512 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0256"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0128"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0032"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=32 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/mles_params.json


# Compare
python -m scenario_gender compare_approaches --output_file "results/scenario_gender__hidden_size.csv" \
    --embedding_file_names \
    "emb__hidden_size_bs_0128_hs_3072.pickle" \
    "emb__hidden_size_bs_0128_hs_2048.pickle" \
    "emb__hidden_size_bs_0128_hs_1024.pickle" \
    "emb__hidden_size_bs_0128_hs_0512.pickle" \
    "emb__hidden_size_bs_0128_hs_0256.pickle" \
    "emb__hidden_size_bs_0128_hs_0128.pickle" \
    "emb__hidden_size_bs_0128_hs_0064.pickle" \
    "emb__hidden_size_bs_0128_hs_0032.pickle"
