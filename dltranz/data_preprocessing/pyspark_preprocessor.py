import logging

from .base import DataPreprocessor

logger = logging.getLogger(__name__)


class PysparkDataPreprocessor(DataPreprocessor):
    def __init__(self, col_id, cols_event_time, cols_category, cols_log_norm, cols_identity, cols_target, print_dataset_info=False):
        super().__init__(col_id, cols_event_time, cols_category, cols_log_norm, cols_identity, cols_target)
        self.print_dataset_info = print_dataset_info

    def _reset(self):
        """Reset internal data-dependent state of the preprocessor, if necessary.
        __init__ parameters are not touched.
        """
        # TODO: pyspark reset data-dependent state of the preprocessor
        pass

    def fit(self, dt, **params):
        """
        Parameters
        ----------
        dt : pandas.DataFrame with flat data

        Returns
        -------
        self : object
            Fitted preprocessor.
        """
        # Reset internal state before fitting
        self._reset()

        # TODO: pyspark fit

        return self

    def transform(self, dt, **params):
        self.check_is_fitted()

        # TODO: pyspark transformation

        return dt

    def check_is_fitted(self):
        # TODO: pyspark is preprocessor fitted check
        pass
