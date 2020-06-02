
for SC_AMOUNT in 496, 992, 1985, 3971, 7943, 15887
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
        params.pretrained_model_path="models/bowl2019_ml_model_ft.p" \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_mles_params.json

    python -m scenario_bowl2019 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.pretrained_model_path="models/cpc_model.p" \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_cpc_params.json

    python -m scenario_bowl2019 pseudo_labeling \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/pseudo_labeling_$SC_AMOUNT"/test \
        output.valid.path="data/pseudo_labeling_$SC_AMOUNT"/valid \
        --conf "conf/dataset.hocon" conf/pseudolabel_params.json

    # Compare
    python -m scenario_bowl2019 compare_approaches \
        --add_baselines --models "lgb" \
        --score_file_names \
            target_scores_$SC_AMOUNT \
            mles_finetuning_scores_$SC_AMOUNT \
            cpc_finetuning_scores_$SC_AMOUNT \
            pseudo_labeling_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file results/semi_scenario_bowl2019_$SC_AMOUNT.csv \
        --embedding_file_names "embeddings.pickle" "embeddings_cpc.pickle"
done


