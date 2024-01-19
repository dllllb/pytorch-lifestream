import time
import pyspark
import pyspark.sql.functions as F
from argparse import Namespace
from unittest import TestCase, main
from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, StructType, StructField

from ptls.make_datasets_spark import DatasetConverter


class DatasetConverterV1(DatasetConverter):
    """Old collect_lists implementation."""
    def collect_lists(self, df, col_id):
        col_list = [col for col in df.columns if col != col_id]

        if self.config.save_partitioned_data:
            df = df.withColumn('mon_id', (F.col('event_time') / 30).cast('int'))
            col_id = [col_id, 'mon_id']

        df = df \
            .withColumn('trx_count', F.count(F.lit(1)).over(Window.partitionBy(col_id))) \
            .withColumn('_rn', F.row_number().over(Window.partitionBy(col_id).orderBy('event_time')))

        for col in col_list:
            df = df.withColumn(col, F.collect_list(col).over(Window.partitionBy(col_id).orderBy('_rn'))) \

        df = df.filter('_rn = trx_count').drop('_rn')
        return df


def cmp_schema(s1, s2, ignore_nullable=False):
    if isinstance(s1, StructType):
        assert isinstance(s2, StructType)
        if len(s1) != len(s2):
            raise AssertionError('The number of columns mismatch')
        s1 = sorted(s1, key=lambda f: f.name)
        s2 = sorted(s2, key=lambda f: f.name)
        for f1, f2 in zip(s1, s2):
            cmp_schema(f1, f2, ignore_nullable=ignore_nullable)
    elif isinstance(s1, StructField):
        if not isinstance(s2, StructField):
            raise AssertionError(f'Different types: {s1} != {s2}.')
        cmp_schema(s1.dataType, s2.dataType, ignore_nullable=ignore_nullable)
        attrs = ['name', 'metadata']
        if not ignore_nullable:
            attrs.append('nullable')
        for attr in attrs:
            if getattr(s1, attr) != getattr(s2, attr):
                raise AssertionError(f'Struct fields differ: {s1} != {s2}.')
    elif isinstance(s1, ArrayType):
        if not isinstance(s2, ArrayType):
            raise AssertionError(f'Different types: {s1} != {s2}.')
        cmp_schema(s1.elementType, s2.elementType, ignore_nullable=ignore_nullable)
        if (not ignore_nullable) and (s1 != s2):
            raise AssertionError(f'Arrays differ: {s1} != {s2}.')
    elif s1 != s2:
        raise AssertionError(f'Elements differ: {s1} != {s2}.')


class TestCollectLists(TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.master("local").appName("test").getOrCreate()
        self.maxDiff = None

    def assertEqualDataframes(self, df1, df2, sort_col, ignore_nullable=False):
        cmp_schema(df1.schema, df2.schema, ignore_nullable=ignore_nullable)
        rows1 = list(map(lambda row: row.asDict(), df1.orderBy(sort_col).collect()))
        rows2 = list(map(lambda row: row.asDict(), df2.orderBy(sort_col).collect()))
        self.assertEqual(len(rows1), len(rows2))
        for row1, row2 in zip(rows1, rows2):
            self.assertEqual(row1, row2)

    def test_compare_v1(self):
        loader = self.spark.read.format("csv").option("header","true").option("inferSchema","true")
        df = loader.load(str(Path(__file__).parent / "age-transactions.csv"))
        expression = (F.col('trans_date') + F.rand()) * 10000
        df = df.withColumn('event_time', expression).drop('trans_date')
        start = time.time()
        converter = DatasetConverter()
        converter_v1 = DatasetConverterV1()
        for save_partitioned_data in [False, True]:
            config = Namespace(save_partitioned_data=save_partitioned_data)
            converter.config = config
            converter_v1.config = config
            df_lists = converter.collect_lists(df, 'client_id')
            df_lists_v1 = converter_v1.collect_lists(df, 'client_id')
            sort_col = ['client_id', 'mon_id'] if save_partitioned_data else 'client_id'
            self.assertEqualDataframes(df_lists, df_lists_v1, sort_col, ignore_nullable=True)


if __name__ == "__main__":
    if 'unittest.util' in __import__('sys').modules:
        # Show full diff in self.assertEqual.
        __import__('sys').modules['unittest.util']._MAX_LENGTH = 999999999

    main()
