import numpy as np


class Dist:
    def sample(self):
        raise NotImplemented


class BetaDist(Dist):
    def __init__(self, a, b, **kwargs):
        self.a = a
        self.b = b

    def sample(self):
        return np.random.beta(self.a, self.b)


class UniformDist(Dist):
    def __init__(self, **kwargs):
        pass

    def sample(self):
        return np.random.rand()


class ConstDist(Dist):
    def __init__(self, n, p, **kwargs):
        assert n == len(p)
        self.n = n
        self.p = p

    def sample(self):
        return np.random.choice(self.n, p=self.p) / self.n


class Feature:
    def sample(self):
        raise NotImplemented


class ConstFeature(Feature):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class CategoryFeature(Feature):
    def __init__(self, n, dist_type='uniform', dist_args={}):
        assert dist_type in ['uniform', 'beta', 'const']
        self.n = n

        if dist_type == 'uniform':
            self.dist = UniformDist()
        elif dist_type == 'beta':
            self.dist = BetaDist(**dist_args)
        elif dist_type == 'const':
            self.dist = ConstDist(n=n, **dist_args)

    def sample(self):
        return int(self.dist.sample() * self.n)


class FloatFeature(Feature):
    def __init__(self, min=0, max=1, log=False, dist_type='uniform', dist_args={}):
        assert dist_type in ['uniform', 'beta']
        self.min = min
        self.max = max
        self.log = log

        if dist_type == 'uniform':
            self.dist = UniformDist()
        elif dist_type == 'beta':
            self.dist = BetaDist(**dist_args)

    def sample(self):
        v = self.min + self.dist.sample() * (self.max - self.min)
        v = np.exp(v) if self.log else v
        return v


class State:
    def __init__(self, features, ind):
        self.features = features
        self.names = set(self.features.keys())
        self.ind = ind

    def unwrap(self):
        return {k: v.sample() for k, v in self.features.items()}


def correct_round(x):
    if x % 1 == 0.5:
        return int(x) + 1
    else:
        return round(x)


@np.vectorize
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def norm_vector(vector):
    if np.all(vector == 0):
        return vector
    else:
        return vector / np.sqrt((vector ** 2).sum())
