
# train special model for fine-tunnig in semi-supervised setup 
# it is quite smaller, than one which is used in supervised setup, due to insufficiency labeled data to train a big model. 
# python ../../metric_learning.py params.device="$SC_DEVICE" \
#     --conf conf/dataset.hocon conf/mles_params_for_finetuning.json

# Train the Contrastive Predictive Coding (CPC) model
# python ../../train_cpc.py    params.device="$SC_DEVICE" \
#   --conf conf/dataset.hocon conf/cpc_params.json

for SC_AMOUNT in 290000 200000 100000 050000 025000 012000 006000 003000 001000 000500
do
	python -m scenario_x5 fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        stats.feature_name="target_scores_${SC_AMOUNT}" \
        stats.path="results/fit_target_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_target_params_rnn.json

    python -m scenario_x5 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="mles_finetuning_${SC_AMOUNT}" \
        stats.path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_mles_params.json

    python -m scenario_x5 fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        stats.path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_cpc_params.json
done

rm results/scenario_x5__semi_supervised.txt
# rm -r conf/embeddings_validation_semi_supervised.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_semi_supervised.hocon --workers 10 --total_cpu_count 20
