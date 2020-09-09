export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="SampleRandom_short"
export SC_STRATEGY="SampleRandom"
python ../../metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.split_strategy.cnt_min=100 \
    params.train.split_strategy.cnt_max=500 \
    params.valid.split_strategy.cnt_min=100 \
    params.valid.split_strategy.cnt_max=500 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_bowl2019 compare_approaches --output_file "results/scenario_bowl2019__subseq_smpl_strategy.csv" \
    --models 'lgb' --embedding_file_names \
    "mles_embeddings.pickle"              \
    "emb__SampleRandom.pickle" \
    "emb__SampleRandom_short.pickle"

