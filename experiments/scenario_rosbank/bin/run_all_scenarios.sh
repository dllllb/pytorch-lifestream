## start with working directory: experiments/scenario_gender
## dataset should be prepared before this script
echo "==== Device ${SC_DEVICE} will be used"


echo ""
echo "==== Main track"
sh bin/scenario_baselines_unsupervised.sh
sh bin/scenario_baselines_supervised.sh


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
