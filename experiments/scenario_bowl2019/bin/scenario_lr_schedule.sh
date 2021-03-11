# ReduceLROnPlateau
export SC_SUFFIX="reduce_on_plateau"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="reduce_on_plateau_x2epochs"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    trainer.max_epochs=200 \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# CosineAnnealing
export SC_SUFFIX="cosine_annealing"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lr_scheduler.n_epoch=150 \
    params.lr_scheduler.CosineAnnealing=true \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/rm results/scenario_lr_schedule.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_lr_schedule.txt",
      auto_features: [
          "../data/emb__reduce_on_plateau.pickle", 
          "../data/emb__reduce_on_plateau_x2epochs.pickle",
          "../data/emb__cosine_annealing.pickle"]'