export SC_SUFFIX="hidden_size_0032"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=32 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=64 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


export SC_SUFFIX="hidden_size_0100"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=100 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size_0200"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=200 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
rm results/scenario_bowl2019__hidden_size.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__hidden_size.txt",
      auto_features: ["../data/emb__hidden_size_*.pickle"]'

