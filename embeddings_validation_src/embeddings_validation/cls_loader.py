def get_cls(cls_name):
    i = cls_name.split('.')
    mod = __import__('.'.join(i[:-1]), fromlist=[i[-1]])
    cls = getattr(mod, i[-1])
    if cls is None:
        raise AttributeError(f'Unknown class name: "{cls_name}"')
    return cls


def create(cls_name, params):
    cls = get_cls(cls_name)
    return cls(**params)
