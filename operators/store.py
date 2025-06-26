import os

from operators.target import Target

def store(target: Target, file: str = None, key: str = None) -> Target:
    class Store(Target):
        def __init__(self):
            super().__init__(key=key)
            self.cache_key = file

        def __call__(self, *args, **kwargs):
            if os.path.exists(self.cache_key):
                with open(self.cache_key, 'r') as f:
                    return f.read()

            result = target(*args, **kwargs)
            with open(self.cache_key, 'w') as f:
                f.write(result)

            return result

    return Store()
