from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer


class ColCategoryTransformer(ColTransformer):
    @property
    def dictionary_size(self):
        raise NotImplementedError()
