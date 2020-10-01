## start with working directory: experiments/scenario_gender
## dataset should be prepared before this script
echo "==== Device ${SC_DEVICE} will be used"


echo ""
echo "==== Main track"
# Prepare agg feature encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json


# Train a supervised model and save scores to the file
python -m scenario_age_pred fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json


# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params_for_finetuning.json
# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../train_cpc.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
# for SC_EPOCH in 04 08 12 16 20 24 28
# do
#     python ../../ml_inference.py \
#         model_path.model="models/cpc_checkpoints/cpc_model_${SC_EPOCH##+(0)}.pth" \
#         output.path="data/cpc_embedding_${SC_EPOCH}" \
#         params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
# done
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
rm results/scenario_age_pred.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation.hocon --workers 10 --total_cpu_count 20


echo ""
echo "==== Hyper parameters tuning"

sh bin/scenario_encoder_type.sh
sh bin/scenario_hidden_size.sh
sh bin/scenario_lr_schedule.sh
sh bin/scenario_ml_loss.sh
sh bin/scenario_sampling_strategy.sh
sh bin/scenario_sub_seq_sampling_strategy.sh
sh bin/scenario_semi_supervised.sh


echo ""
echo "==== Other scenarios"

# sh bin/scenario_projection_head.sh
