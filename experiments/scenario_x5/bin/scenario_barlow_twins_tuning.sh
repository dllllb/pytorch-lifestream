export SC_SUFFIX="bt_tuning_base"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=256 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=3 \
    trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v01"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=1024 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=5 \
    trainer.max_epochs=150 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v02"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=1024 \
    params.train.lr=0.001 \
    params.lr_scheduler.step_size=5 \
    trainer.max_epochs=150 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v03"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=512 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=3 \
    trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"


export SC_SUFFIX="bt_tuning_v01"; export SC_VERSION=4
export SC_SUFFIX="bt_tuning_v02"; export SC_VERSION=1

ls "lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/"
# ep = 0; st = 282; {i: (st + 1) // (ep + 1) * (i + 1) - 1 for i in range(ep, 600, 1)}

python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=0-step\=282.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_000" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=9-step\=2829.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_009" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=19-step\=5659.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_019" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=29-step\=8489.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_029" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=39-step\=11319.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_039" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=49-step\=14149.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_049" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=59-step\=16979.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_059" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=69-step\=19809.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_069" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=79-step\=22639.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_079" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=89-step\=25469.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_089" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=1000 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=99-step\=28299.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_099" \
    --conf "conf/barlow_twins_params.hocon"


rm results/res_bt_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 4 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_tuning.txt",
      auto_features: ["../data/emb__bt_tuning_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_tuning.txt

