# GRU encoder
export SC_SUFFIX="encoder_gru"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="gru" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_4head_64hs_4layers"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.transf.n_heads=4 \
    params.transf.n_layers=4 \
    params.train.batch_size=64 \
    params.valid.batch_size=64 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=32 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_8head_64hs_4layers"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.transf.n_heads=8 \
    params.transf.n_layers=4 \
    params.train.batch_size=64 \
    params.valid.batch_size=64 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=32 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_4head_64hs_8layers"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.transf.n_heads=4 \
    params.transf.n_layers=8 \
    params.train.batch_size=64 \
    params.valid.batch_size=64 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=32 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_bowl2019 compare_approaches --models 'lgb' --output_file "results/scenario_bowl2019__encoder_types.csv" \
    --embedding_file_names \
    "mles_embeddings.pickle"              \
    "emb__encoder_gru.pickle"                                \
    "emb__encoder_lstm.pickle"                               \
    "emb__encoder_transf_bs064_4head_64hs_4layers.pickle"         \
    "emb__encoder_transf_bs064_8head_64hs_4layers.pickle"         \
    "emb__encoder_transf_bs064_4head_64hs_8layers.pickle"



