
for SC_AMOUNT in 496 994 1986 3971 7943 15887
do
	python -m scenario_bowl2019 fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_target_params.json

    python -m scenario_bowl2019 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.frooze_trx_encoder=true \
        params.train.n_epoch=15 \
        params.train.lr_scheduler.step_gamma=0.5 \
        params.train.lr_scheduler.step_size=5 \
        params.train.lr=0.01 \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_mles_params.json

    python -m scenario_bowl2019 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=15 \
        params.train.lr_scheduler.step_gamma=0.1 \
        params.train.lr_scheduler.step_size=10 \
        params.train.lr=0.001 \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_cpc_params.json

    # Compare
    python -m scenario_bowl2019 compare_approaches \
        --add_baselines --models "lgb" \
        --score_file_names \
            target_scores_$SC_AMOUNT \
            mles_finetuning_scores_$SC_AMOUNT \
            cpc_finetuning_scores_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file results/semi_scenario_bowl2019_$SC_AMOUNT.csv

done


