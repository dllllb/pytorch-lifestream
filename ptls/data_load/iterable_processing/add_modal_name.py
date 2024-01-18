from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np

class AddModalName(IterableProcessingDataset):
    '''Add_Modal_Name(cols = ['mcc', 'amount'], source = 'Source1') ---> Source1_mcc, Source1_amount'''
    def __init__(self, cols, source):
        super().__init__()
        self._cols = cols
        self._source = source
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            for feature in self._cols:
                if feature in features:
                    features[self._source + '_' + feature] = features[feature]
                    del features[feature]
            yield rec