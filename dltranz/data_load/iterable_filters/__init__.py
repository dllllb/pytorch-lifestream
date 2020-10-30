class FilterChain:
    def __init__(self, *i_filters):
        self.i_filters = i_filters

    def __call__(self, seq):
        for f in self.i_filters:
            seq = f(seq)
        return seq
