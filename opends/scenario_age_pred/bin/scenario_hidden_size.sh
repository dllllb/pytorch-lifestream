# HiddenSize for batch_size=512
export SC_SUFFIX="hidden_size_bs_0512_hs_224"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=224 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_192"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=192 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_160"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=160 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=128 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_096"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=96 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=64 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_032"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=32 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_016"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=16 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

# HiddenSize for batch_size=256
export SC_SUFFIX="hidden_size_bs_0256_hs_480"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=480 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_352"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=352 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_224"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=224 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_160"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=160 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

# HiddenSize for batch_size=1024
export SC_SUFFIX="hidden_size_bs_1024_hs_096"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=96 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_1024_hs_064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=64 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# Compare
python -m scenario_age_pred compare_approaches --output_file "runs/scenario_age_pred__encoder_types.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__hidden_size_bs_0512_hs_224.pickle"  \
    "emb__hidden_size_bs_0512_hs_192.pickle"  \
    "emb__hidden_size_bs_0512_hs_160.pickle"  \
    "emb__hidden_size_bs_0512_hs_128.pickle"  \
    "emb__hidden_size_bs_0512_hs_096.pickle"  \
    "emb__hidden_size_bs_0512_hs_064.pickle"  \
    "emb__hidden_size_bs_0512_hs_032.pickle"  \
    "emb__hidden_size_bs_0512_hs_016.pickle"  \
    "emb__hidden_size_bs_0256_hs_480.pickle"  \
    "emb__hidden_size_bs_0256_hs_352.pickle"  \
    "emb__hidden_size_bs_0256_hs_224.pickle"  \
    "emb__hidden_size_bs_0256_hs_160.pickle"  \
    "emb__hidden_size_bs_1024_hs_096.pickle"  \
    "emb__hidden_size_bs_1024_hs_064.pickle"
