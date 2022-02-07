## start with working directory: experiments/scenario_gender
## dataset should be prepared before this script
echo "==== Folds split"
rm -r lightning_logs/
rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20 \
    --split_only

echo "==== Device cuda:${CUDA_VISIBLE_DEVICES} will be used"


echo ""
echo "==== Main track"
sh bin/scenario_baselines_unsupervised.sh
#sh bin/scenario_baselines_supervised.sh


#echo ""
#echo "==== Hyper parameters tuning"

#sh bin/scenario_encoder_type.sh
#sh bin/scenario_hidden_size.sh
#sh bin/scenario_lr_schedule.sh
#sh bin/scenario_ml_loss.sh
#sh bin/scenario_sampling_strategy.sh
#sh bin/scenario_sub_seq_sampling_strategy.sh
#sh bin/scenario_semi_supervised.sh


#echo ""
#echo "==== Other scenarios"
