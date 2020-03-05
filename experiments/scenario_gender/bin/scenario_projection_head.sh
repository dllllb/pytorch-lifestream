export SC_SUFFIX="projection_head_rnn0256_prh256"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=256 \
    params.projection_head.output_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json


export SC_SUFFIX="projection_head_rnn0512_prh256"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=512 \
    params.projection_head.output_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn1024_prh256"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=1024 \
    params.projection_head.output_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn1024_prh128"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=1024 \
    params.projection_head.output_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn0512_prh128"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=512 \
    params.projection_head.output_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn0256_prh128"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=256 \
    params.projection_head.output_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn1024_prh064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=1024 \
    params.projection_head.output_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn0512_prh064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=512 \
    params.projection_head.output_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn0256_prh064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=256 \
    params.projection_head.output_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="projection_head_rnn0128_prh064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=256 \
    params.projection_head.output_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/dataset.hocon conf/gender_ml_params_inference.json

# Compare
python -m scenario_gender compare_approaches --output_file "results/scenario_gender__projection_head.csv" \
    --embedding_file_names \
    "emb__projection_head_rnn0256_prh256.pickle" \
    "emb__projection_head_rnn0512_prh256.pickle" \
    "emb__projection_head_rnn1024_prh256.pickle" \
    "emb__projection_head_rnn1024_prh128.pickle" \
    "emb__projection_head_rnn0512_prh128.pickle" \
    "emb__projection_head_rnn0256_prh128.pickle" \
    "emb__projection_head_rnn1024_prh064.pickle" \
    "emb__projection_head_rnn0512_prh064.pickle" \
    "emb__projection_head_rnn0256_prh064.pickle" \
    "emb__projection_head_rnn0128_prh064.pickle"

