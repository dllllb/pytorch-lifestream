from sklearn.preprocessing import OneHotEncoder


class CategoryEncoder:
    """Check input dataframe, search categories, end encode them
    Drop very frequent categories
    """

    def __init__(
            self,
            keep_dtype_kinds=('f', 'i'),
            max_features=10,
            max_unique_rate=0.8,
            min_coverage_rate=0.2,
    ):
        self.keep_dtype_kinds = keep_dtype_kinds
        self.max_features = max_features
        self.max_unique_rate = max_unique_rate
        self.min_coverage_rate = min_coverage_rate

        self.cols_for_encoding = None
        self.cols_for_encoding_info = None
        self.cols_for_drop = None
        self.encoder = None

    def fit_cols_for_encoding(self, df):
        self.cols_for_encoding = []

        for col, dtype in df.dtypes.items():
            if dtype.kind in self.keep_dtype_kinds:
                continue
            self.cols_for_encoding.append(col)

    def fit_cols_for_drop(self, df):
        self.cols_for_drop = {}

        for col in self.cols_for_encoding:
            n_unique = df[col].nunique()
            if n_unique == 1:
                self.cols_for_drop[col] = 'Droped with n_unique == 1'

            unique_rate = n_unique / len(df)
            if unique_rate >= self.max_unique_rate:
                self.cols_for_drop[col] = 'Droped with unique_rate >= max_unique_rate ' + \
                                          f'({unique_rate:.3f} >= {self.max_unique_rate:.3f})'

            coverage_rate = df[col].value_counts(normalize=True).cumsum().iloc[:self.max_features].iloc[-1]
            if coverage_rate < self.min_coverage_rate:
                self.cols_for_drop[col] = 'Droped with coverage_rate < min_coverage_rate ' + \
                                          f'({coverage_rate:.3f} < {self.min_coverage_rate:.3f})'

        self.cols_for_encoding = [c for c in self.cols_for_encoding if c not in self.cols_for_drop]

    def fit_transform(self, df):
        self.fit_cols_for_encoding(df)
        self.fit_cols_for_drop(df)
        self.cols_for_encoding_info = {}

        _df = df[self.cols_for_encoding].astype(str)
        categories = []
        for col in _df.columns:
            v = _df[col].value_counts().iloc[:self.max_features]
            self.cols_for_encoding_info[col] = f'{len(v)} values with {v.sum() / len(_df):.2} coverage'
            v = v.index.tolist()
            v = sorted(v)
            categories.append(v)

        self.encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False)
        self.encoder.fit(df[self.cols_for_encoding].astype(str))
        return self.transform(df)

    def transform(self, df):
        if len(self.cols_for_drop) > 0:
            df = df.drop(columns=self.cols_for_drop)

        values = self.encoder.transform(df[self.cols_for_encoding].astype(str))

        for i in range(values.shape[1]):
            df[f'_encoded__{i}'] = values[:, i]
        df = df.drop(columns=self.cols_for_encoding)

        return df
