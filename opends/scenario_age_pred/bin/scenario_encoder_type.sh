# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json
#
# GRU encoder
export SC_SUFFIX="encoder_gru"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json
#
#
# Transformer encoder
export SC_SUFFIX="encoder_transf_bs512_2head_064hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.transf.n_heads=2 \
    params.transf.input_size=64 \
    params.transf.dim_hidden=64 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs512_2head_032hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.transf.n_heads=2 \
    params.transf.input_size=32 \
    params.transf.dim_hidden=32 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs512_2head_064hs_2layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.transf.n_heads=2 \
    params.transf.input_size=64 \
    params.transf.dim_hidden=64 \
    params.transf.n_layers=2 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs512_2head_32hs_2layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.transf.n_heads=2 \
    params.transf.input_size=32 \
    params.transf.dim_hidden=32 \
    params.transf.n_layers=2 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs256_4head_128hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=256 \
    params.transf.n_heads=4 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs256_2head_128hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=256 \
    params.transf.n_heads=2 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs256_4head_064hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=256 \
    params.transf.n_heads=4 \
    params.transf.input_size=64 \
    params.transf.dim_hidden=64 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs256_4head_128hs_2layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=256 \
    params.transf.n_heads=4 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=2 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_4head_196hs_4layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=128 \
    params.transf.n_heads=4 \
    params.transf.input_size=196 \
    params.transf.dim_hidden=196 \
    params.transf.n_layers=4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="encoder_transf_bs128_4head_196hs_6layers"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.split_strategy.cnt_max=150 \
    params.train.batch_size=128 \
    params.transf.n_heads=4 \
    params.transf.input_size=196 \
    params.transf.dim_hidden=196 \
    params.transf.n_layers=6 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# Compare
python -m scenario_age_pred compare_approaches --output_file "runs/scenario_age_pred__encoder_types.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__encoder_lstm.pickle"                                 \
    "emb__encoder_gru.pickle"                                  \
    "emb__encoder_transf_bs512_2head_064hs_4layers.pickle"     \
    "emb__encoder_transf_bs512_2head_032hs_4layers.pickle"     \
    "emb__encoder_transf_bs512_2head_064hs_2layers.pickle"     \
    "emb__encoder_transf_bs512_2head_032hs_2layers.pickle"     \
    "emb__encoder_transf_bs256_4head_128hs_4layers.pickle"     \
    "emb__encoder_transf_bs256_2head_128hs_4layers.pickle"     \
    "emb__encoder_transf_bs256_4head_064hs_4layers.pickle"     \
    "emb__encoder_transf_bs256_4head_128hs_2layers.pickle"     \
    "emb__encoder_transf_bs128_4head_196hs_4layers.pickle"     \
    "emb__encoder_transf_bs128_4head_196hs_6layers.pickle"

