export SC_EPOCH_N=1

export SC_SUFFIX="hidden_size__hs_0064"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.rnn.hidden_size=64 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="hidden_size__hs_0160"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.rnn.hidden_size=160 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
	
export SC_SUFFIX="hidden_size__hs_0480"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.train.batch_size=128 \
    params.rnn.hidden_size=480 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
	
export SC_SUFFIX="hidden_size__hs_0800"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.train.batch_size=64 \
    params.rnn.hidden_size=800 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
    
export SC_SUFFIX="hidden_size__hs_1600"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.n_epoch=$SC_EPOCH_N \
    params.train.batch_size=64 \
    params.rnn.hidden_size=1600 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__hidden_size.csv" \
    --embedding_file_names \
    "emb__hidden_size__hs_1600.pickle" \
    "emb__hidden_size__hs_0800.pickle"  \
    "emb__hidden_size__hs_0480.pickle"  \
    "emb__hidden_size__hs_0160.pickle"  \
    "emb__hidden_size__hs_0064.pickle"
