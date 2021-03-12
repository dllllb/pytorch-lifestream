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
        
        self.source_df = df

        return df
    
    def load_target(self):
        df_target = self.load_source_data(self.config.target_files)
        df_target = df_target.select([self.config.col_client_id, self.config.col_target[0]])
        
        # Filter & Merge with source dataframe
        filtered_df = (
            self.source_df
            .where((F.col('event_type') == 'Assessment') & (F.col('event_code') == 2000))
            .select(['installation_id', self.config.col_client_id, self.config.cols_event_time[0]])
        )
        df_target = df_target.join(filtered_df, on=[self.config.col_client_id], how='left')
        
        return df_target

    def trx_to_features(self, df_data, print_dataset_info,
                        col_client_id, cols_event_time, cols_category, cols_log_norm, max_trx_count):
        encoders = {col: self.get_encoder(df_data, col) for col in cols_category}
        for col in cols_category:
            df_data = self.encode_col(df_data, col, encoders[col])

        used_columns = cols_category + ['event_time', 'installation_id']
        features = df_data.select(used_columns)
        features = self.remove_long_trx(features, max_trx_count, 'installation_id')
        features = self.collect_lists(features, 'installation_id')
        features.persist()

        return features
    
    def update_with_target(self, features, df_target, col_client_id, col_target):
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

if __name__ == '__main__':
    LocalDatasetConverter().run()
