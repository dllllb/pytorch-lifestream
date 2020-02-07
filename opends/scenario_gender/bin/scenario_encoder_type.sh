# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_lstm_hs128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_lstm_short"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    params.train.split_strategy.cnt_min=15 \
    params.train.split_strategy.cnt_max=75 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


export SC_SUFFIX="encoder_lstm_long"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    params.train.split_strategy.cnt_min=100 \
    params.train.split_strategy.cnt_max=300 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


#
# GRU encoder
export SC_SUFFIX="encoder_gru"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


export SC_SUFFIX="encoder_gru_hs128"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    params.rnn.hidden_size=128 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_gru_short"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    params.train.split_strategy.cnt_min=15 \
    params.train.split_strategy.cnt_max=75 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_gru_long"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    params.train.split_strategy.cnt_min=100 \
    params.train.split_strategy.cnt_max=300 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
#
#
# Transformer encoder
export SC_SUFFIX="encoder_transf_bs128_8head_128hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=8 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_4head_128hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=4 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_8head_128hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=8 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=4 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_8head_096hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=96 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_8head_064hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=8 \
    params.transf.input_size=64 \
    params.transf.dim_hidden=64 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs064_8head_192hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=64 \
    params.transf.n_heads=8 \
    params.transf.input_size=192 \
    params.transf.dim_hidden=192 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs064_8head_128hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=64 \
    params.transf.n_heads=8 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_4head_128hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.n_heads=4 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=4 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=64 \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# Compare
python -m scenario_gender compare_approaches --output_file "runs/scenario_gender__encoder_types.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__encoder_lstm.pickle"                               \
    "emb__encoder_lstm_hs128.pickle"                         \
    "emb__encoder_lstm_short.pickle"                         \
    "emb__encoder_lstm_long.pickle"                          \
    "emb__encoder_gru.pickle"                                \
    "emb__encoder_gru_hs128.pickle"                          \
    "emb__encoder_gru_short.pickle"                          \
    "emb__encoder_gru_long.pickle"                           \
    "emb__encoder_transf_bs128_8head_128hs_6layers.pickle"   \
    "emb__encoder_transf_bs128_4head_128hs_6layers.pickle"   \
    "emb__encoder_transf_bs128_8head_128hs_4layers.pickle"   \
    "emb__encoder_transf_bs128_8head_096hs_6layers.pickle"   \
    "emb__encoder_transf_bs128_8head_064hs_6layers.pickle"   \
    "emb__encoder_transf_bs064_8head_192hs_6layers.pickle"   \
    "emb__encoder_transf_bs064_8head_128hs_6layers.pickle"   \
    "emb__encoder_transf_bs128_4head_128hs_4layers.pickle"
