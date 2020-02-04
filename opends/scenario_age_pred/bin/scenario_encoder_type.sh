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

# # Transformer encoder
# python -m scenario_age_pred fit_target \
#     params.device="$SC_DEVICE" \
#     params.model_type="transf" \
#     params.train.SubsamplingDataset.enabled=false \
#     params.valid.batch_size=32 \
#     output.valid.path="../data/age-pred/ts__encoder_transf/valid" \
#     output.test.path="../data/age-pred/ts__encoder_transf/test" \
#     --conf conf/age_pred_dataset.hocon conf/age_pred_target_params_train.json


# Compare
python -m scenario_age_pred compare_approaches --output_file "runs/scenario_age_pred__encoder_types.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__encoder_lstm.pickle" \
    "emb__encoder_gru.pickle"



