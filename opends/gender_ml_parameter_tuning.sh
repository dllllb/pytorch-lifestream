python metric_learning.py model_path.model="models/gender_ml_model_v_base.p"      params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_hs512.p"     params.rnn.hidden_size=512 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_epoch250.p"  params.rnn.hidden_size=256 params.train.n_epoch=250 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_bs32.p"      params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=32  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_bs128.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=128 params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_bs256.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=256 params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_len150.p"    params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=75  params.train.split_strategy.cnt_max=200 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_len60.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=75  --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python metric_learning.py model_path.model="models/gender_ml_model_v_len250.p"    params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=200 params.train.split_strategy.cnt_max=300 --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json

python ml_inference.py model_path.models=["models/gender_ml_model_v_base.p"    ] output.path="../data/gender/embeddings_v_base"     --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_hs512.p"   ] output.path="../data/gender/embeddings_v_hs512"    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_epoch250.p"] output.path="../data/gender/embeddings_v_epoch250" --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_bs32.p"    ] output.path="../data/gender/embeddings_v_bs32"     --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_bs128.p"   ] output.path="../data/gender/embeddings_v_bs128"    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_bs256.p"   ] output.path="../data/gender/embeddings_v_bs256"    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_len150.p"  ] output.path="../data/gender/embeddings_v_len150"   --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_len60.p"   ] output.path="../data/gender/embeddings_v_len60"    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json
python ml_inference.py model_path.models=["models/gender_ml_model_v_len250.p"  ] output.path="../data/gender/embeddings_v_len250"   --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

python -m scenario_gender compare_approaches --pos 5 6 7 8 9 10 11 12 13 --ml_embedding_file_names \
    "embeddings_v_base.pickle"     \
    "embeddings_v_hs512.pickle"    \
    "embeddings_v_epoch250.pickle" \
    "embeddings_v_bs32.pickle"     \
    "embeddings_v_bs128.pickle"    \
    "embeddings_v_bs256.pickle"    \
    "embeddings_v_len150.pickle"   \
    "embeddings_v_len60.pickle"    \
    "embeddings_v_len250.pickle"

#                     (rocauc_score, mean)  (rocauc_score, std)       (rocauc_score, <lambda_0>) metric_learning_embedding_name
#     pos model_type
#     5   linear                    0.8624               0.0091  [0.850 0.856 0.867 0.870 0.870]       embeddings_v_base.pickle
#         xgb                       0.8578               0.0108  [0.843 0.854 0.857 0.865 0.871]       embeddings_v_base.pickle
#     6   linear                    0.8608               0.0083  [0.848 0.859 0.861 0.866 0.870]      embeddings_v_hs512.pickle
#         xgb                       0.8581               0.0075  [0.850 0.851 0.859 0.863 0.868]      embeddings_v_hs512.pickle
#     7   linear                    0.8631               0.0097  [0.850 0.859 0.862 0.871 0.873]   embeddings_v_epoch250.pickle
#         xgb                       0.8575               0.0101  [0.841 0.857 0.858 0.863 0.868]   embeddings_v_epoch250.pickle
#     8   linear                    0.8573               0.0106  [0.845 0.851 0.856 0.862 0.872]       embeddings_v_bs32.pickle
#         xgb                       0.8521               0.0096  [0.838 0.849 0.852 0.857 0.864]       embeddings_v_bs32.pickle
#     9   linear                    0.8606               0.0091  [0.848 0.855 0.863 0.866 0.871]      embeddings_v_bs128.pickle
#         xgb                       0.8531               0.0065  [0.844 0.849 0.853 0.859 0.860]      embeddings_v_bs128.pickle
#     10  linear                    0.8625               0.0074  [0.855 0.855 0.865 0.865 0.872]      embeddings_v_bs256.pickle
#         xgb                       0.8570               0.0095  [0.845 0.849 0.862 0.864 0.865]      embeddings_v_bs256.pickle
#     11  linear                    0.8550               0.0075  [0.842 0.855 0.857 0.860 0.861]     embeddings_v_len150.pickle
#         xgb                       0.8464               0.0064  [0.836 0.845 0.850 0.851 0.851]     embeddings_v_len150.pickle
#     12  linear                    0.8565               0.0084  [0.843 0.854 0.860 0.861 0.865]      embeddings_v_len60.pickle
#         xgb                       0.8539               0.0089  [0.840 0.854 0.854 0.858 0.864]      embeddings_v_len60.pickle
#     13  linear                    0.8426               0.0097  [0.828 0.837 0.845 0.850 0.852]     embeddings_v_len250.pickle
#         xgb                       0.8423               0.0134  [0.826 0.832 0.845 0.850 0.859]     embeddings_v_len250.pickle


python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_base.p"     output.path="../data/gender/finetuning_scores_v_base"       --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_hs512.p"    output.path="../data/gender/finetuning_scores_v_hs512"    params.rnn.hidden_size=512 --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_epoch250.p" output.path="../data/gender/finetuning_scores_v_epoch250"   --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_bs32.p"     output.path="../data/gender/finetuning_scores_v_bs32"       --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_bs128.p"    output.path="../data/gender/finetuning_scores_v_bs128"      --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_bs256.p"    output.path="../data/gender/finetuning_scores_v_bs256"      --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_len150.p"   output.path="../data/gender/finetuning_scores_v_len150"     --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_len60.p"    output.path="../data/gender/finetuning_scores_v_len60"      --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json
python -m scenario_gender fit_finetuning params.pretrained_model_path="models/gender_ml_model_v_len250.p"   output.path="../data/gender/finetuning_scores_v_len250"     --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

python -m scenario_gender compare_approaches --pos 6 7 8 9 10 11 12 13 14 --target_score_file_names \
    "finetuning_scores_v_base"     \
    "finetuning_scores_v_hs512"    \
    "finetuning_scores_v_epoch250" \
    "finetuning_scores_v_bs32"     \
    "finetuning_scores_v_bs128"    \
    "finetuning_scores_v_bs256"    \
    "finetuning_scores_v_len150"   \
    "finetuning_scores_v_len60"    \
    "finetuning_scores_v_len250"

