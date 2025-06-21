from operators.target import Target

def cache(target: Target):
    class Cache(Target):
        def __init__(self):
            super().__init__()
            self.cache = None

        def __call__(self, *args, **kwargs):
            if self.cache is not None:
                return self.cache
            self.cache = target(*args, **kwargs)
            return self.cache

    return Cache()

def test_cache():
    pass
