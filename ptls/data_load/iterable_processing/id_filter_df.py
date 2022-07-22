from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class IdFilterDf(IterableProcessingDataset):
    def __init__(self, df_relevant_ids):
        """Remove records which are not in `relevant_ids`
        Use possible multicolumn dataframe as `relevant_ids`

        Args:
            df_relevant_ids: list of ids which should be kept
        """
        super().__init__()

        self._df_relevant_ids = set(tuple(r) for r in df_relevant_ids.values.tolist())
        self.id_columns = df_relevant_ids.columns
        self.id_types = [type(v) for v in next(iter(self._df_relevant_ids))]

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            _id = tuple([col_type(features[col]) for col, col_type in zip(self.id_columns, self.id_types)])
            if _id not in self._df_relevant_ids:
                continue
            yield rec
