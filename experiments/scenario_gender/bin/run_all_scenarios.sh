## start with working directory: experiments/scenario_gender
## dataset should be prepared before this script
echo "==== Device ${SC_DEVICE} will be used"


echo ""
echo "==== Main track"
# Prepare agg feature encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json


# Train a supervised model and save scores to the file
python -m scenario_gender fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
# Fine tune the MeLES model in supervised mode and save scores to the file

python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.rnn.hidden_size=256 \
  model_path.model="models/mles_model_for_finetuning.p" \
  --conf conf/dataset.hocon conf/mles_params.json
python -m scenario_gender fit_finetuning \
  params.device="$SC_DEVICE" \
  --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json


# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../train_cpc.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_gender compare_approaches --n_workers 5 --models lgb \
    --output_file results/scenario_gender.csv \
    --baseline_name "agg_feat_embed.pickle" \
    --embedding_file_names "mles_embeddings.pickle" "cpc_embeddings.pickle" \
    --score_file_names "target_scores" "mles_finetuning_scores" "cpc_finetuning_scores"


echo ""
echo "==== Hyper parameters tuning"

sh bin/scenario_encoder_type.sh
sh bin/scenario_hidden_size.sh
sh bin/scenario_lr_schedule.sh
sh bin/scenario_ml_loss.sh
sh bin/scenario_sampling_strategy.sh
sh bin/scenario_sub_seq_sampling_strategy.sh

sh bin/scenario_semi_supervised_with_embedd_validation.sh

echo ""
echo "==== Other scenarios"

sh bin/scenario_projection_head.sh
