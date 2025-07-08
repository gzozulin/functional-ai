from concurrent.futures import ThreadPoolExecutor

from operators.dummy import dummy
from operators.target import Target

def fork(target: Target, mapper, reducer, key: str = None) -> Target:
    class Fork(Target):
        def __init__(self):
            super().__init__(key=key)
            self._target = target
            self._mapper = mapper
            self._reducer = reducer

        def __call__(self, *args, **kwargs):
            result = {self._target.key: self._target(*args, **kwargs)}
            targets = mapper(**result)

            with ThreadPoolExecutor() as executor:
                results = [executor.submit(trg, *args, **kwargs) for trg in targets]
                results = [f.result() for f in results]

            return self._reducer(results)

    return Fork()

def test_fork():
    def example_mapper(dum):
        return [dummy(f"Mapped 1: {dum}"),
                dummy(f"Mapped 2: {dum}")]

    def example_reducer(mapped_results):
        return " | ".join(mapped_results)

    forked_target = fork(
        target=dummy("Hello, World!", key="dum"),
        mapper=example_mapper,
        reducer=example_reducer,
        key="forked_example"
    )

    assert forked_target() == "Mapped 1: Hello, World! | Mapped 2: Hello, World!"
