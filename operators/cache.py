from operators.target import Target

def cache(target: Target, key: str = None) -> Target:
    class Cache(Target):
        def __init__(self):
            super().__init__(key=key)
            self.cache = None
            self.cache_key = None

        def _update_cache_key(self, **kwargs):
            cache_key = hash(tuple(sorted(kwargs.items())))
            if self.cache_key is None or self.cache_key != cache_key:
                self.cache = None
                self.cache_key = cache_key

        def __call__(self, *args, **kwargs):
            self._update_cache_key(**kwargs)

            if self.cache is not None:
                return self.cache

            self.cache = target(*args, **kwargs)
            return self.cache

    return Cache()

def test_cache():
    pass
