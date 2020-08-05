export SC_SUFFIX="sample_01_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="sample_02_mles"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy="SampleSlicesMLES" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="sample_02_mles_p"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy="SampleSlicesMLES" \
    params.train.split_strategy.short_seq_crop_rate=0.8 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="sample_02_sync"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy="SampleSlices" \
    dataset.min_seq_len=5 \
    params.train.split_strategy.cnt_min=10 \
    params.train.split_strategy.cnt_max=800 \
    params.train.batch_size=128 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="sample_03_mles"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy="SampleSlicesMLES" \
    dataset.min_seq_len=5 \
    params.train.split_strategy.cnt_min=10 \
    params.train.split_strategy.cnt_max=800 \
    params.train.batch_size=128 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

export SC_SUFFIX="sample_04_mles_p"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy="SampleSlicesMLES" \
    dataset.min_seq_len=5 \
    params.train.split_strategy.cnt_min=10 \
    params.train.split_strategy.cnt_max=800 \
    params.train.batch_size=128 \
    params.train.split_strategy.short_seq_crop_rate=0.8 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

python -m scenario_x5 compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_embeddings_sample_*.pickle"


#                                                     oof_accuracy                                                         test_accuracy
#                                                             mean t_int_l t_int_h    std                           values          mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: mles_embeddings_sample_01_base.pickle         0.5425  0.5405  0.5444 0.0016  [0.541 0.541 0.543 0.544 0.544]        0.5423  0.5407  0.5439 0.0013  [0.541 0.541 0.542 0.543 0.544]
# lgb_embeds: mles_embeddings_sample_02_mles.pickle         0.5451  0.5430  0.5472 0.0017  [0.543 0.544 0.545 0.546 0.547]        0.5434  0.5423  0.5445 0.0009  [0.542 0.543 0.544 0.544 0.544]
# lgb_embeds: mles_embeddings_sample_02_mles_p.pickle       0.5453  0.5432  0.5475 0.0018  [0.543 0.544 0.545 0.547 0.547]        0.5436  0.5415  0.5457 0.0017  [0.542 0.542 0.543 0.545 0.546]
# lgb_embeds: mles_embeddings_sample_02_sync.pickle         0.5394  0.5366  0.5423 0.0023  [0.536 0.538 0.540 0.541 0.542]        0.5363  0.5346  0.5380 0.0014  [0.534 0.535 0.537 0.537 0.538]
# lgb_embeds: mles_embeddings_sample_03_mles.pickle         0.5419  0.5401  0.5437 0.0015  [0.541 0.541 0.541 0.543 0.544]        0.5409  0.5399  0.5420 0.0009  [0.540 0.540 0.541 0.542 0.542]
