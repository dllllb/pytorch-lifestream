# Train the MeLES encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/mles_params.hocon
python ../../pl_inference.py --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=96 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_01"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_02"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=64 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_03"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=32 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_04"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=24 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_05"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=256 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_05_1"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=256 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0003 \
    params.lr_scheduler.params.total_steps=3600 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_05_p"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=256 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0003 \
    params.lr_scheduler.params.total_steps=3600 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_06_2"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=2048 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=1 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_06_3"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=2048 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=1 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0001 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_07"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=512 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.001 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_08"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=2048 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=1 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0003 \
    params.lr_scheduler.params.total_steps=3600 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_09"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=1024 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=1 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0003 \
    params.lr_scheduler.params.total_steps=3600 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_10"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=2048 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=1 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0003 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0003 \
    params.lr_scheduler.params.total_steps=3600 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_params.hocon


export SC_SUFFIX="encoder_transf_11"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=256 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.001 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.001 \
    params.lr_scheduler.params.total_steps=6000   trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=10 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_12"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=256 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    params.train.lr=0.0001 \
    params.lr_scheduler.type="OneCycleLR" \
    params.lr_scheduler.params.max_lr=0.0001 \
    params.lr_scheduler.params.total_steps=6000   trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=10 \
    params.lr_scheduler.params.pct_start=0.15 \
    params.lr_scheduler.params.anneal_strategy="cos" \
    params.lr_scheduler.params.cycle_momentum=False \
    params.lr_scheduler.params.div_factor=20 \
    params.lr_scheduler.params.final_div_factor=200 \
    params.lr_scheduler.params.three_phase=true \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon


# ls -l lightning_logs/${SC_SUFFIX}/version_0/checkpoints/
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=9-step\=589.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0009" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=19-step\=1179.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0019" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=29-step\=1769.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0029" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=39-step\=2359.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0039" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=49-step\=2949.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0049" \
    --conf conf/mles_params.hocon

CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=59-step\=3539.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0059" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=69-step\=4129.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0069" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=79-step\=4719.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0079" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=89-step\=5309.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0089" \
    --conf conf/mles_params.hocon
CUDA_VISIBLE_DEVICES=3 python ../../pl_inference.py \
    model_path="lightning_logs/${SC_SUFFIX}/version_0/checkpoints/epoch\=99-step\=5899.ckpt" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__${SC_SUFFIX}_0099" \
    --conf conf/mles_params.hocon

export SC_SUFFIX="encoder_transf_mlm_10"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

export SC_SUFFIX="encoder_transf_mlm_11"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

export SC_SUFFIX="encoder_transf_mlm_12"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.mlm_loss.loss_mlm_w=0.0 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

export SC_SUFFIX="encoder_transf_mlm_13"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.mlm_loss.loss_mlm_w=0.0 \
    params.train.mlm_loss.loss_var_w=0.0 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

export SC_SUFFIX="encoder_transf_mlm_14"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.mlm_loss.loss_mlm_w=0.0 \
    params.train.mlm_loss.loss_var_w=0.0 \
    params.train.mlm_loss.replace_prob=0.01 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

export SC_SUFFIX="encoder_transf_mlm_15"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=512 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=true \
    params.transf.use_src_key_padding_mask=false \
    params.train.mlm_loss.loss_mlm_w=1.0 \
    params.train.mlm_loss.loss_var_w=1.0 \
    params.train.mlm_loss.replace_prob=0.1 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf conf/mles_transf_params.hocon
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=64 \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf conf/mles_transf_params.hocon

# Compare
rm results/scenario_rosbank__transf_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank__transf_tuning.txt",
      auto_features: ["../data/emb_mles__encoder_*.pickle"]'
