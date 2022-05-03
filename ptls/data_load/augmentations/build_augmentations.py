from ptls.data_load import augmentation_chain


def build_augmentations(conf):
    def _chain():
        from .all_time_shuffle import AllTimeShuffle
        from .dropout_trx import DropoutTrx
        from .random_slice import RandomSlice
        from .seq_len_limit import SeqLenLimit
        from .drop_day import DropDay

        for cls_name, params in conf:
            cls_f = locals().get(cls_name)
            if cls_f is None:
                raise AttributeError(f'Can not find augmentation for "{cls_name}"')
            yield cls_f(**params)

    return augmentation_chain(*_chain())
