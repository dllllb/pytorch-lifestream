# Train the MeLES encoder and take embedidngs; inference
# python ../../pl_train_module.py --conf conf/mles_params.hocon
# python ../../pl_inference.py --conf conf/mles_params.hocon

# Train the Transformers embedder; inference
python ../../pl_train_module.py --conf conf/transformer_params.hocon
python ../../pl_inference.py --conf conf/transformer_params.hocon


# Compare
rm -f results/scenario_age_pred_transformer.txt

rm -rf conf/embeddings_validation.work/

python -m embeddings_validation \
   --conf conf/embeddings_validation_transformer.hocon --workers 10 --total_cpu_count 20
