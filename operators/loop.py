from operators import dummy
from operators.target import Target

def loop(target: Target, condition, reducer, key: str = None):
    class Loop(Target):
        def __init__(self):
            super().__init__(key=key)
            self.target = target
            self.condition = condition
            self.reducer = reducer

        def __call__(self, *args, **kwargs):
            results = []

            index = 0
            while condition(idx=index, **kwargs):
                result = self.target(*args, **kwargs, idx=index)
                results.append(result)
                kwargs[target.key] = result
                index += 1

            return self.reducer(results)

    return Loop()

def loopn(target: Target, count, key: str = None):
    return loop(target, lambda idx: idx < count, key)

def test_loop():
    def target_func(idx):
        return f"Iteration {idx}"

    def reducer(results):
        return " | ".join(results)

    loop_target = loop(
        target=dummy(template=target_func),
        condition=lambda idx, **kwargs: idx < 5,
        reducer=reducer
    )

    result = loop_target()
    assert result == "Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4", \
        "Loop should iterate 5 times and return the correct results"
