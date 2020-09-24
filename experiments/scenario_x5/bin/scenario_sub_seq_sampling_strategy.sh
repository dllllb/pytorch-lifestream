export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.batch_size=128 \
    params.valid.batch_size=128 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.batch_size=128 \
    params.valid.batch_size=128 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__subseq_smpl_strategy.csv" \
    --models lgb \
    --embedding_file_names \
    "mles_embeddings.pickle"                    \
    "emb__SplitRandom.pickle" \
    "emb__SampleRandom.pickle"
