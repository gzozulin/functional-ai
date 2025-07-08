import os

from operators import dummy
from operators.target import Target

class Storage:
    def __init__(self):
        self._storage = None

    def get(self):
        return self._storage

    def set(self, value):
        self._storage = value

    def clear(self):
        self._storage = None

    def is_empty(self):
        return self._storage is None

class FileStorage(Storage):
    def __init__(self, filename: str):
        super().__init__()
        self._filename = filename

    def get(self):
        with open(self._filename, 'r') as f:
            return f.read()

    def set(self, value):
        with open(self._filename, 'w') as f:
            f.write(value)

    def clear(self):
        if os.path.exists(self._filename):
            os.remove(self._filename)

    def is_empty(self):
        return not os.path.exists(self._filename)

class Cache(Target):
    def __init__(self, key, target: Target, storage: Storage):
        super().__init__(key=key)
        self._storage = storage
        self._target = target

    def clear(self):
        self._storage.clear()

    def __call__(self, *args, **kwargs):
        if self._storage.is_empty():
            self._storage.set(self._target(*args, **kwargs))

        return self._storage.get()

def cache(target: Target, key: str = None) -> Cache:
    return Cache(target=target, key=key, storage=Storage())

def store(target: Target, filename: str, key: str = None) -> Cache:
    return Cache(target=target, key=key, storage=FileStorage(filename=filename))

def test_cache():
    c = cache(target=dummy("test"))
    assert c(a=1) == "test", "Cache should return the cached value on first call"
    assert c(a=2) == "test", "Cache should return the same value on subsequent calls"

def test_store():
    filename = "test_cache.txt"
    if os.path.exists(filename):
        os.remove(filename)

    s = store(target=dummy("test"), filename=filename)
    assert s(a=1) == "test", "Store should return the value on first call"

    with open(filename, 'r') as f:
        assert f.read() == "test", "Stored value should match the target output"

    os.remove(filename)

