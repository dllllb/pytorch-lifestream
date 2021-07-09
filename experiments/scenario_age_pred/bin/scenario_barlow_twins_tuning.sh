
export SC_SUFFIX="bt_tuning_lambd_0.020-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.080-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.08 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0600-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=600 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_1000-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1000 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_256-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=256 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0003-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.0003 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0030-weight_decay_0.000-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.003 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.001-step_size_30-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0.001 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_20-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=20 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_40-step_gamma_0.9025"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=40 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.7"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.7 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"
export SC_SUFFIX="bt_tuning_lambd_0.040-hidden_size_0800-prj_size_000-batch_size_128-lr_0.0010-weight_decay_0.000-step_size_30-step_gamma_0.97"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=30 \
    params.lr_scheduler.step_gamma=0.97 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"


#####
export SC_SUFFIX="bt_tuning_ep600_lr0.0003"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=800 \
    data_module.train.batch_size=128 \
    params.train.lr=0.0003 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=60 \
    params.lr_scheduler.step_gamma=0.97 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_ep600_prj256_lr0.0008_hs1600"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1600 \
    "params.head_layers=[[Linear, {in_features: 1600, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256}], [ReLU, {}], [Linear, {in_features: 256, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.0008 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=60 \
    params.lr_scheduler.step_gamma=0.97 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"


export SC_SUFFIX="bt_tuning_ep600_lr0.0003"
export SC_SUFFIX="bt_tuning_ep600_prj256"
export SC_SUFFIX="bt_tuning_ep600_prj256_lr0.0008_hs1600"
export SC_VERSION=1
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=9-step\=3489.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_009" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=19-step\=6979.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_019" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=29-step\=10469.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_029" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=39-step\=13959.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_039" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=49-step\=17449.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_049" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=59-step\=20939.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_059" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=69-step\=24429.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_069" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=79-step\=27919.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_079" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=89-step\=31409.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_089" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=99-step\=34899.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_099" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=109-step\=38389.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_109" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=119-step\=41879.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_119" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=129-step\=45369.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_129" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=139-step\=48859.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_139" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=149-step\=52349.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_149" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=199-step\=69799.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_199" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=249-step\=87249.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_249" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=299-step\=104699.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_299" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=349-step\=122149.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_349" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=399-step\=139599.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_399" \
    --conf "conf/barlow_twins_params.hocon"

rm results/res_bt_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_tuning.txt",
      auto_features: ["../data/emb__bt_tuning_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_tuning.txt


'epoch=109-step=38389.ckpt'  'epoch=199-step=69799.ckpt'   'epoch=29-step=10469.ckpt'    'epoch=389-step=136109.ckpt'  'epoch=479-step=167519.ckpt'  'epoch=569-step=198929.ckpt'                 │+-------------------------------+----------------------+----------------------+
'epoch=119-step=41879.ckpt'  'epoch=209-step=73289.ckpt'   'epoch=299-step=104699.ckpt'  'epoch=39-step=13959.ckpt'    'epoch=489-step=171009.ckpt'  'epoch=579-step=202419.ckpt'                 │|   3  Tesla P100-PCIE...  On   | 0000CFBB:00:00.0 Off |                  Off |
'epoch=129-step=45369.ckpt'  'epoch=219-step=76779.ckpt'   'epoch=309-step=108189.ckpt'  'epoch=399-step=139599.ckpt'  'epoch=49-step=17449.ckpt'    'epoch=589-step=205909.ckpt'                 │| N/A   23C    P0    25W / 250W |     10MiB / 16280MiB |      0%      Default |
'epoch=139-step=48859.ckpt'  'epoch=229-step=80269.ckpt'   'epoch=319-step=111679.ckpt'  'epoch=409-step=143089.ckpt'  'epoch=499-step=174499.ckpt'  'epoch=59-step=20939.ckpt'                   │+-------------------------------+----------------------+----------------------+
'epoch=149-step=52349.ckpt'  'epoch=239-step=83759.ckpt'   'epoch=329-step=115169.ckpt'  'epoch=419-step=146579.ckpt'  'epoch=509-step=177989.ckpt'  'epoch=599-step=209399.ckpt'                 │
'epoch=159-step=55839.ckpt'  'epoch=249-step=87249.ckpt'   'epoch=339-step=118659.ckpt'  'epoch=429-step=150069.ckpt'  'epoch=519-step=181479.ckpt'  'epoch=69-step=24429.ckpt'                   │+-----------------------------------------------------------------------------+
'epoch=169-step=59329.ckpt'  'epoch=259-step=90739.ckpt'   'epoch=349-step=122149.ckpt'  'epoch=439-step=153559.ckpt'  'epoch=529-step=184969.ckpt'  'epoch=79-step=27919.ckpt'                   │| Processes:                                                       GPU Memory |
'epoch=179-step=62819.ckpt'  'epoch=269-step=94229.ckpt'   'epoch=359-step=125639.ckpt'  'epoch=449-step=157049.ckpt'  'epoch=539-step=188459.ckpt'  'epoch=89-step=31409.ckpt'                   │|  GPU       PID   Type   Process name                             Usage      |
'epoch=189-step=66309.ckpt'  'epoch=279-step=97719.ckpt'   'epoch=369-step=129129.ckpt'  'epoch=459-step=160539.ckpt'  'epoch=549-step=191949.ckpt'  'epoch=9-step=3489.ckpt'                     │|=============================================================================|
'epoch=19-step=6979.ckpt'    'epoch=289-step=101209.ckpt'  'epoch=379-step=132619.ckpt'  'epoch=469-step=164029.ckpt'  'epoch=559-step=195439.ckpt'  'epoch=99-step=34899.ckpt'                   │|    0     13544      C   /mnt2/molchanov/.venv/bin/python             825MiB |



