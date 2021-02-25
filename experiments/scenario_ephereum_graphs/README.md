
# pytorch_lightning framework

```sh
cd experiments/scenario_ephereum_graphs

#  `conf/dataset_iterable_file.hocon` may be included in `conf/mles_params.hocon`

# CoLES unsupervised
python ../../pl_train_module.py \
     trainer.gpus=[0] \
     --conf conf/mles_params.hocon

cd ../../
python pl_inference_spark.py \
    work_path="/mnt/ildar" \
    spark_memory="32G" \
    --conf experiments/scenario_ephereum_graphs/conf/mles_params.hocon

```
