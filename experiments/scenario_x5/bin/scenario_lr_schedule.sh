# ReduceLROnPlateau
export SC_SUFFIX="reduce_on_plateau"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.patience=3 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="reduce_on_plateau_x2epochs"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    params.lr_scheduler.patience=3 \
    params.train.n_epoch=60 \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# CosineAnnealing
export SC_SUFFIX="cosine_annealing"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.CosineAnnealing=true \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_lr_schedule.csv" \
    --n_workers 2 --models lgb --embedding_file_names \
    "mles_embeddings.pickle"        \
    "emb__reduce_on_plateau.pickle" \
    "emb__reduce_on_plateau_x2epochs.pickle" \
    "emb__cosine_annealing.pickle"
