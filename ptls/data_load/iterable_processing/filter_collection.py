from functools import partial
from ptls.data_load.iterable_processing.filtering import Filtering


CategorySizeClip = partial(Filtering, mode='CategorySizeClip')
SeqLenFilter = partial(Filtering, mode='SeqLenFilter')
DeleteNan = partial(Filtering, mode='DeleteNan')
IdFilter = partial(Filtering, mode='IdFilter')
ISeqLenLimit = partial(Filtering, mode='ISeqLenLimit')
FilterNonArray = partial(Filtering, mode='FilterNonArray')
IdFilterDf = partial(Filtering, mode='IdFilterDf')
