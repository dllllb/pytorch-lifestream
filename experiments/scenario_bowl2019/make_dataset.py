from make_datasets_spark import DatasetConverter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType, StringType, IntegerType
from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
import os
import json
import logging
import datetime
import numpy as np


logger = logging.getLogger(__name__)

class LocalDatasetConverter(DatasetConverter):
    def load_transactions(self):
        file_name_train, file_name_test = self.config.trx_files

        df_train = self.spark_read_file(self.path_to_file(file_name_train))
        df_test = self.spark_read_file(self.path_to_file(file_name_test))

        logger.info(f'Loaded {df_train.count()} records from "{file_name_train}"')
        logger.info(f'Loaded {df_test.count()} records from "{file_name_test}"')

        for col in df_train.columns:
            if col not in df_test.columns:
                df_test = df_test.withColumn(col, F.lit(None))
                logger.info(f'Test extended with "{col}" column')

        df = df_train.union(df_test)

        # timestamp to float
        frmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        col_event_time = self.config.cols_event_time[0]
        df = df.withColumn(col_event_time, F.unix_timestamp(col_event_time, frmt))
        df = df.withColumn(col_event_time, F.col(col_event_time) / (24 * 60 * 60))
        df = df.withColumn('event_time', df[col_event_time])
        
        # Process key == 'correct' in json data
        udf_function = udf(lambda x: str(json.loads(x).get('correct', 'None')), StringType())
        df = df.withColumn('correct', udf_function('event_data'))
        
        # Delete useless columns
        df = df.drop('event_data', 'event_count')

        return df
    
    def load_target(self, source_df):
        df_target = self.load_source_data(self.config.target_files)
        df_target = df_target.select([self.config.col_client_id, self.config.col_target[0]])
        
        # Filter & Merge with source dataframe
        filtered_df = (
            source_df
            .where((F.col('event_type') == 'Assessment') & (F.col('event_code') == 2000))
            .select(['installation_id', self.config.col_client_id, self.config.cols_event_time[0]])
        )
        df_target = df_target.join(filtered_df, on=[self.config.col_client_id], how='left')
        
        return df_target

    def update_with_target(self, features, df_target):
        data = df_target.join(features, on=['installation_id'], how='left')
        
        # Find index for find last timestamp in event_time sequences
        def get_index(event_time, timestamp):
            return int(np.searchsorted(np.array(event_time), timestamp)) + 1

        udf_function = udf(get_index, IntegerType())
        data = data.withColumn('index', udf_function('event_time', self.config.cols_event_time[0]))

        # Slice transactions  by index
        cols_to_slice = ['event_id', 'event_code', 'event_type','title', 'world', 'correct']
        for col in cols_to_slice:
            udf_function = udf(lambda seq, index: seq[0: index], ArrayType(IntegerType()))
            data = data.withColumn(col, udf_function(col, 'index'))

        udf_function = udf(lambda seq, index: seq[0: index], ArrayType(DoubleType()))
        data = data.withColumn('event_time', udf_function('event_time', 'index'))
        
        # Update trx_count since transaction were cutted by index 
        udf_function = udf(lambda seq: len(seq), IntegerType())
        data = data.withColumn('trx_count', udf_function('event_time'))
        
        # Remove useless columns
        data = data.drop('index', self.config.cols_event_time[0])

        return data

    def run(self):
        _start = datetime.datetime.now()
        self.parse_args()
        spark = SparkSession.builder.getOrCreate()

        self.logging_config()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'load_source_data')
        source_data = self.load_transactions()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'trx_to_features')
        client_features = self.trx_to_features(
            df_data=source_data,
            print_dataset_info=self.config.print_dataset_info,
            col_client_id='installation_id',
            cols_event_time=self.config.cols_event_time,
            cols_category=self.config.cols_category,
            cols_log_norm=self.config.cols_log_norm,
            max_trx_count=self.config.max_trx_count,
        )

        # load target
        df_target = self.load_target(source_data)
        df_target.persist()
        col_target = self.config.col_target[0]

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'update_with_target')
        client_features = self.update_with_target(features=client_features, df_target=df_target)

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'split_dataset')
        train, test, save_test_id = None, None, False
        train, test = self.split_dataset(
            all_data=client_features,
            test_size=float(self.config.test_size),
            df_target=df_target,
            col_client_id=self.config.col_client_id,
            salt=self.config.salt,
        )
        save_test_id = True

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'save_features')
        self.save_features(
            df_data=train,
            save_path=self.config.output_train_path,
        )

        self.save_features(
            df_data=test,
            save_path=self.config.output_test_path,
        )

        test_ids = test.select(self.config.col_client_id).distinct().toPandas()
        test_ids.to_csv(self.config.output_test_ids_path, index=False)

        _duration = datetime.datetime.now() - _start
        logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

if __name__ == '__main__':
    LocalDatasetConverter().run()
