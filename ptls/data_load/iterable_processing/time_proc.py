from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np

class TimeProc(IterableProcessingDataset):
    '''This class is used for generation weekday and hour features from time_col feature'''
    def __init__(self, time_col):
        super().__init__()
        self._time_col = time_col
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if self._time_col in features:
                ts = np.array(features[self._time_col]).astype('datetime64[s]')
                day = np.array(ts).astype('datetime64[D]')

                features['weekday'] = (day.view('int64') - 4) % 7 + 1
                features['hour'] = (ts - day).astype('timedelta64[h]').view('int64')
            yield rec

class TimeProcMultimodal(IterableProcessingDataset):
    '''This class is used for generation weekday and hour features from used source'''
    def __init__(self, time_col, source):
        super().__init__()
        self._time_col = time_col
        self._source = source
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if self._time_col in features:
                ts = np.array(features[self._time_col]).astype('datetime64[s]')
                day = np.array(ts).astype('datetime64[D]')

                features[self._source + '_weekday'] = (day.view('int64') - 4) % 7 + 1
                features[self._source + '_hour'] = (ts - day).astype('timedelta64[h]').view('int64')
            yield rec