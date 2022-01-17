from typing import List


class DataPreprocessor:
    def __init__(self,
                 col_id: str,
                 cols_event_time: str,
                 cols_category: List[str],
                 cols_log_norm: List[str]):
        self.col_id = col_id
        self.cols_event_time = cols_event_time
        self.cols_category = cols_category
        self.cols_log_norm = cols_log_norm
        self._reset()

    def _reset(self):
        """Reset internal data-dependent state of the preprocessor, if necessary.
        __init__ parameters are not touched.
        """
        self.cols_category_mapping = {}

    def check_is_fitted(self):
        """Check is preprocessor is fitted, before applying transformation
        """
        assert all(k in self.cols_category_mapping for k in self.cols_category), "Preprocessor is not fitted"

    def fit(self, dt, **params):
        raise NotImplementedError()

    def transform(self, dt, **params):
        raise NotImplementedError()

    def fit_transform(self, dt, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `dt` with optional parameters `fit_params`
        and returns a transformed version of `dt`.
        Parameters
        ----------
        dt : Input samples.
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        dt_transformed : Transformed data.
        """
        return self.fit(dt, **fit_params).transform(dt)
