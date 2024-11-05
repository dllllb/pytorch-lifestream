from weakref import WeakValueDictionary
from dask.distributed import Client, LocalCluster

class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DaskServer(metaclass=Singleton):
    def __init__(self):
        print('Creating Dask Server')
        cluster = LocalCluster(processes=False,
                               n_workers=4,
                               threads_per_worker=4,
                               memory_limit='auto'
                               )
        # connect client to your cluster
        self.client = Client(cluster)
