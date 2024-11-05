from ptls.data_load.iterable_processing.feature_preprocessing import FeaturePreprocessing
from functools import partial


FeatureBinScaler = partial(FeaturePreprocessing, mode='FeatureBinScaler')
FeatureFilter = partial(FeaturePreprocessing, mode='FeatureFilter')
FeatureRename = partial(FeaturePreprocessing, mode='FeatureRename')
FeatureTypeCast = partial(FeaturePreprocessing, mode='FeatureTypeCast')
