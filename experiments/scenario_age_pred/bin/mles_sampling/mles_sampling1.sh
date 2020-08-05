# # # # # # # # # # # # # # # #
### MELES sampling vs old sampling
# # # # # # # # # # # # # # # #


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
export SC_SUFFIX="base"

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
    params.train.split_strategy.cnt_max=1200 \
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
    params.train.split_strategy.cnt_max=1200 \
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
    params.train.split_strategy.cnt_max=1200 \
    params.train.split_strategy.short_seq_crop_rate=0.8 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    output.path="data/mles_embeddings_${SC_SUFFIX}" \
    --conf conf/dataset.hocon conf/mles_params.json

python -m scenario_age_pred compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_embeddings_sample_*.pickle"

#                                                     oof_accuracy                                                         test_accuracy
#                                                             mean t_int_l t_int_h    std                           values          mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: mles_embeddings_sample_01_base.pickle         0.6372  0.6324  0.6419 0.0038  [0.632 0.636 0.636 0.641 0.641]        0.6419  0.6377  0.6461 0.0034  [0.636 0.642 0.643 0.644 0.645]
# lgb_embeds: mles_embeddings_sample_02_mles.pickle         0.6356  0.6272  0.6440 0.0068  [0.627 0.633 0.636 0.636 0.646]        0.6393  0.6362  0.6423 0.0024  [0.637 0.637 0.639 0.641 0.642]
# lgb_embeds: mles_embeddings_sample_02_mles_p.pickle       0.6356  0.6272  0.6440 0.0068  [0.627 0.633 0.636 0.636 0.646]        0.6393  0.6362  0.6423 0.0024  [0.637 0.637 0.639 0.641 0.642]
# lgb_embeds: mles_embeddings_sample_02_sync.pickle         0.5666  0.5600  0.5732 0.0053  [0.558 0.566 0.568 0.571 0.571]        0.5625  0.5570  0.5681 0.0045  [0.555 0.562 0.563 0.565 0.567]
# lgb_embeds: mles_embeddings_sample_03_mles.pickle         0.6145  0.6123  0.6168 0.0018  [0.612 0.614 0.615 0.615 0.617]        0.6173  0.6143  0.6202 0.0024  [0.615 0.615 0.619 0.619 0.619]
# lgb_embeds: mles_embeddings_sample_04_mles_p.pickle       0.6145  0.6123  0.6168 0.0018  [0.612 0.614 0.615 0.615 0.617]        0.6173  0.6143  0.6202 0.0024  [0.615 0.615 0.619 0.619 0.619]
