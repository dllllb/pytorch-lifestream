# Base batch_size=64, hidden_size=256

# HiddenSize for batch_size=1024
export SC_SUFFIX="hidden_size_bs_1024_hs_0448"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=448 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_1024_hs_0384"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=384 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_1024_hs_0320"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=320 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_1024_hs_0256"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=1024 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# HiddenSize for batch_size=512
export SC_SUFFIX="hidden_size_bs_0512_hs_0512"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=512 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0448"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=448 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0384"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=384 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0320"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=320 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0256"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0192"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=192 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0032"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=32 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0512_hs_0016"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=512 \
    params.rnn.hidden_size=16 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# HiddenSize for batch_size=256
export SC_SUFFIX="hidden_size_bs_0256_hs_1024"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=1024 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_0768"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=768 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_0512"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=512 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_0256"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

#######
export SC_SUFFIX="hidden_size_bs_0256_hs_0128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0256_hs_0064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=256 \
    params.rnn.hidden_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# HiddenSize for batch_size=128
export SC_SUFFIX="hidden_size_bs_0128_hs_3072"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=3072 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_2048"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=2048 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_1024"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=1024 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0512"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=512 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0256"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0032"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=128 \
    params.rnn.hidden_size=32 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


# HiddenSize for batch_size=64
export SC_SUFFIX="hidden_size_bs_0064_hs_4096"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=4096 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_3072"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=3072 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_2048"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=2048 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    params.valid.batch_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_1024"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=1024 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_0512"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=512 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_0256"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_0128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0064_hs_0064"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.batch_size=64 \
    params.rnn.hidden_size=64 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# LR correction
export SC_SUFFIX="hidden_size_bs_0128_hs_0256_lr_0.004"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.lr=0.004 \
    params.train.batch_size=128 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="hidden_size_bs_0128_hs_0256_lr_0.001"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.lr=0.001 \
    params.train.batch_size=128 \
    params.rnn.hidden_size=256 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# Compare
python -m scenario_gender compare_approaches --output_file "runs/scenario_gender__hidden_size.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__hidden_size_bs_1024_hs_0448.pickle" \
    "emb__hidden_size_bs_1024_hs_0384.pickle" \
    "emb__hidden_size_bs_1024_hs_0320.pickle" \
    "emb__hidden_size_bs_1024_hs_0256.pickle" \
    "emb__hidden_size_bs_0512_hs_0512.pickle" \
    "emb__hidden_size_bs_0512_hs_0448.pickle" \
    "emb__hidden_size_bs_0512_hs_0384.pickle" \
    "emb__hidden_size_bs_0512_hs_0320.pickle" \
    "emb__hidden_size_bs_0512_hs_0256.pickle" \
    "emb__hidden_size_bs_0512_hs_0192.pickle" \
    "emb__hidden_size_bs_0512_hs_0128.pickle" \
    "emb__hidden_size_bs_0512_hs_0064.pickle" \
    "emb__hidden_size_bs_0512_hs_0032.pickle" \
    "emb__hidden_size_bs_0512_hs_0016.pickle" \
    "emb__hidden_size_bs_0256_hs_1024.pickle" \
    "emb__hidden_size_bs_0256_hs_0768.pickle" \
    "emb__hidden_size_bs_0256_hs_0512.pickle" \
    "emb__hidden_size_bs_0256_hs_0256.pickle" \
    "emb__hidden_size_bs_0256_hs_0128.pickle" \
    "emb__hidden_size_bs_0256_hs_0064.pickle" \
    "emb__hidden_size_bs_0128_hs_3072.pickle" \
    "emb__hidden_size_bs_0128_hs_2048.pickle" \
    "emb__hidden_size_bs_0128_hs_1024.pickle" \
    "emb__hidden_size_bs_0128_hs_0512.pickle" \
    "emb__hidden_size_bs_0128_hs_0256.pickle" \
    "emb__hidden_size_bs_0128_hs_0128.pickle" \
    "emb__hidden_size_bs_0128_hs_0064.pickle" \
    "emb__hidden_size_bs_0064_hs_4096.pickle" \
    "emb__hidden_size_bs_0064_hs_3072.pickle" \
    "emb__hidden_size_bs_0064_hs_2048.pickle" \
    "emb__hidden_size_bs_0064_hs_1024.pickle" \
    "emb__hidden_size_bs_0064_hs_0512.pickle" \
    "emb__hidden_size_bs_0064_hs_0256.pickle" \
    "emb__hidden_size_bs_0064_hs_0128.pickle" \
    "emb__hidden_size_bs_0064_hs_0064.pickle"
