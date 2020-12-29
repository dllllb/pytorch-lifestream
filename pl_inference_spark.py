import logging
import torch
import pytorch_lightning as pl
from dltranz.lightning_modules.coles_module import CoLESModule
from dltranz.lightning_modules.cpc_module import CpcModule
from dltranz.lightning_modules.rtd_module import RtdModule
from dltranz.lightning_modules.sop_nsp_module import SopNspModule
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from dltranz.util import get_conf

logger = logging.getLogger(__name__)

class InferenceSpark(object):

    def __init__(self, spark, work_path, model_path, output_path, dataset_files, 
                 col_id, num_workers, pl_module_name, hidden_size):
        self.spark          = spark
        self.work_path      = work_path
        self.model_path     = model_path
        self.output_path    = output_path
        self.dataset_files  = dataset_files
        self.col_id         = col_id
        self.num_workers    = num_workers
        self.pl_module_name = pl_module_name
        self.hidden_size    = hidden_size

    def collect_batchs(self):

        df_t = self.spark.read.parquet(*[f"{self.work_path}/{i}" for i in self.dataset_files])

        columns = [i for i in df_t.columns if i not in [self.col_id, "trx_count", "target"]]

        df_t\
        .orderBy("trx_count")\
        .withColumn("group_id",F.monotonically_increasing_id())\
        .groupby(F.floor(F.col("group_id")/350).alias("group_id"))\
        .agg(
            F.collect_list(self.col_id).alias(self.col_id),
            F.struct([F.collect_list(F.col(name)).alias(name) for name in columns]).alias("feature_arrays"),
            F.collect_list("trx_count").alias("trx_count"))\
        .repartition(1)\
        .write.format("parquet").mode("overwrite")\
        .save(f"{self.work_path}/data/tmp_batch_train")

    def exec_inference(self):

        pl.seed_everything(42)

        pl_module = None
        for m in [CoLESModule, CpcModule, SopNspModule, RtdModule]:
            if m.__name__ == self.pl_module_name:
                pl_module = m
                break
        if pl_module is None:
            raise NotImplementedError(f'Unknown pl module {self.pl_module_name}')

        model = pl_module.load_from_checkpoint(f"{self.work_path}/{self.model_path}")
        model.seq_encoder.is_reduce_sequence = True

        br_m = self.spark.sparkContext.broadcast(model.seq_encoder)

        def inference_func(data_feature, data_length, num_cuda):

            import torch
            from dltranz.trx_encoder import PaddedBatch
            
            if torch.cuda.is_available():
                core = f"cuda:{num_cuda}"
            else:
                core = "cpu"
                
            device = torch.device(core)

            data_obj = PaddedBatch(
                payload={
                    k: torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(x) for x in v], batch_first=True
                    ).to(device) for k, v in data_feature.asDict().items()
                },
                length=torch.IntTensor(data_length).to(device)
            )

            br_m.value.to(device)
            br_m.value.eval()

            with torch.no_grad():
                outputs = br_m.value(data_obj)
            torch.cuda.empty_cache()
            return outputs.cpu().numpy().tolist()

        inference_func_udf = F.udf(inference_func, T.ArrayType(T.ArrayType(T.FloatType())))

        self.spark.read.parquet(f"{self.work_path}/data/tmp_batch_train")\
        .repartition(100)\
        .select(
            F.col(self.col_id),
            inference_func_udf(
                F.col("feature_arrays"),
                F.col("trx_count"),
                F.col("group_id")%self.num_workers
            ).alias("inf_res")
        )\
        .repartition(1)\
        .write.format("parquet").mode("overwrite")\
        .save(f"{self.work_path}/data/tmp_res_emb")

    def explode_embedd(self):

        self.spark.read.parquet(f"{self.work_path}/data/tmp_res_emb")\
        .withColumn("cols_zip",F.arrays_zip(self.col_id,"inf_res"))\
        .withColumn("cols_explode",F.explode("cols_zip"))\
        .select(F.col("cols_explode")[self.col_id].alias(self.col_id),
            *[F.col("cols_explode")["inf_res"][i].alias(f"v{i}") for i in range(self.hidden_size)])\
        .repartition(1)\
        .write.format("parquet").mode("overwrite")\
        .save(f"{self.work_path}/{self.output_path}")

def main(args=None):

    conf = get_conf(args)

    spark = SparkSession.builder\
        .appName("spark_inference")\
        .master(f"local[{conf.inference_dataloader.loader.num_workers}]")\
        .config("spark.sql.shuffle.partitions",50)\
        .enableHiveSupport()\
        .getOrCreate()

    inference_obj = InferenceSpark(spark, 
        conf['work_path'], 
        conf['model_path'], 
        conf['output.path'], 
        conf['inference_dataloader.dataset_files'],
        conf['inference_dataloader.col_id'],
        conf["inference_dataloader.loader.num_workers"],
        conf['params.pl_module_name'],
        conf['params.rnn.hidden_size']
        )

    inference_obj.collect_batchs()
    inference_obj.exec_inference()
    inference_obj.explode_embedd()

    spark.stop()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s : %(message)s')
    logging.getLogger("spark_inference").setLevel(logging.INFO)
    main()
