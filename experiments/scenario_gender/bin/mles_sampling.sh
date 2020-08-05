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

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_embeddings_sample_*.pickle"


#                                                     oof_rocauc_score                                                         test_rocauc_score
#                                                                 mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: mles_embeddings_sample_01_base.pickle             0.8690  0.8597  0.8783 0.0075  [0.858 0.865 0.871 0.873 0.878]            0.8821  0.8803  0.8838 0.0014  [0.880 0.882 0.882 0.882 0.884]
# lgb_embeds: mles_embeddings_sample_02_mles.pickle             0.8722  0.8680  0.8764 0.0034  [0.868 0.869 0.874 0.875 0.875]            0.8730  0.8672  0.8789 0.0047  [0.866 0.872 0.873 0.875 0.879]
# lgb_embeds: mles_embeddings_sample_02_mles_p.pickle           0.8722  0.8680  0.8764 0.0034  [0.868 0.869 0.874 0.875 0.875]            0.8730  0.8672  0.8789 0.0047  [0.866 0.872 0.873 0.875 0.879]
# lgb_embeds: mles_embeddings_sample_02_sync.pickle             0.8609  0.8558  0.8660 0.0041  [0.856 0.857 0.861 0.864 0.866]            0.8681  0.8632  0.8730 0.0039  [0.865 0.866 0.866 0.870 0.874]
# lgb_embeds: mles_embeddings_sample_03_mles.pickle             0.8712  0.8614  0.8809 0.0078  [0.859 0.870 0.871 0.876 0.880]            0.8812  0.8785  0.8838 0.0021  [0.879 0.880 0.880 0.881 0.885]
# lgb_embeds: mles_embeddings_sample_04_mles_p.pickle           0.8690  0.8591  0.8788 0.0080  [0.856 0.869 0.869 0.872 0.878]            0.8816  0.8778  0.8855 0.0031  [0.878 0.880 0.882 0.882 0.886]
