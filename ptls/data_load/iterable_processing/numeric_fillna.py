from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np
import warnings

class NumericFillNa(IterableProcessingDataset):
    '''
      Replace NaN with zero and infinity with large finite numbers (default behaviour) or 
      with the numbers defined by the user using the fill_by, posinf and/or neginf keywords.
    '''
    def __init__(self, cols, fill_by=0, posinf=None, neginf=None):
        super().__init__()
        self._cols = cols if cols else []
        self._fill_by = fill_by
        self._posinf = posinf
        self._neginf = neginf
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            for feature in self._cols:
                if np.isnan(features[feature]).any():
                    warnings.warn(f"Warning: There is nan values in column '{feature}'. It will be replaced by {self._fill_by}")
                features[feature] = np.nan_to_num(np.array(features[feature]).astype('float32').view('float32'),
                                                  nan=self._fill_by,
                                                  posinf=self._posinf,
                                                  neginf=self._neginf
                                                 )
            yield rec