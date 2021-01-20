# export SC_DEVICE="cuda"

# step I. Preparing aggregated features dataset with timestamps
python ../../agg_features_ts_preparation.py params.device="$SC_DEVICE" \
    --conf conf/dataset.hocon conf/agg_features_timestamps.json

# step II. Prepare agg features
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_timestamps.json

# step III. Train an unsupervised encoder for aggregated features; inference
python ../../features_encoding.py --conf conf/agg_features_encoding_params.json
python ../../ml_inference.py \
    model_path.model="models/agg_features_encoder.p" \
    output.path="data/agg_feat_embed_encoded" \
    --conf conf/dataset.hocon conf/agg_features_timestamps.json

# step IV. Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__encoding_agg_features.csv" \
    --n_workers 1 --models lgb --embedding_file_names \
    "agg_feat_embed_before.pickle"         \
    "agg_feat_embed_encoded.pickle"


