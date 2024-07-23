from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer


class FrequencyTransformer(ColCategoryTransformer):
    @property
    def dictionary_size(self):
        raise NotImplementedError()
