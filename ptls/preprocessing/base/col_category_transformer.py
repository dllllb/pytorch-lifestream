from ptls.preprocessing.base import ColTransformer


class ColCategoryTransformer(ColTransformer):
    @property
    def dictionary_size(self):
        raise NotImplementedError()
