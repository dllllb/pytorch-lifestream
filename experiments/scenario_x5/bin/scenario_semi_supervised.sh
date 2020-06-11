
# train special model for fine-tunnig in semi-supervised setup 
# it is quite smaller, than one which is used in supervised setup, due to insufficiency labeled data to train a big model. 
python ../../metric_learning.py params.device="$SC_DEVICE" \
    --conf conf/dataset.hocon conf/mles_params_for_finetuning.json

# Train the Contrastive Predictive Coding (CPC) model
python ../../train_cpc.py    params.device="$SC_DEVICE" \
  --conf conf/dataset.hocon conf/cpc_params.json

for SC_AMOUNT in 290000 200000 100000 50000 25000 12000 6000 3000 1000 500
do
	python -m scenario_x5 fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_target_params_rnn.json

    python -m scenario_x5 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_mles_params.json

    python -m scenario_x5 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_cpc_params.json

    # Compare
    python -m scenario_x5 compare_approaches \
        --baseline_name "agg_feat_embed.pickle" --models "lgb" \
        --score_file_names \
            target_scores_$SC_AMOUNT \
            mles_finetuning_scores_$SC_AMOUNT \
            cpc_finetuning_scores_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file results/semi_scenario_x5_$SC_AMOUNT.csv
done


