from concurrent.futures import ThreadPoolExecutor

from operators.infer import infer
from operators.dummy import dummy
from operators.target import Target

def parallel(targets: list[Target], reducer, key: str = None):
    class Parallel(Target):
        def __init__(self):
            super().__init__(key=key)
            self.targets = targets
            self.reducer = reducer

        def __call__(self, *args, **kwargs):
            with ThreadPoolExecutor() as executor:
                results = [executor.submit(target, *args, **kwargs) for target in targets]
                results = [f.result() for f in results]
                results = {target.key: result for target, result in zip(self.targets, results)}

            return self.reducer(**kwargs, **results)

    return Parallel()

def ai_parallel(template, targets: list[Target], llm: str = None, tools: list = None, key: str = None):
    def reducer(**kwargs):
        return infer(template, llm, tools)(**kwargs)
    return parallel(targets, reducer, key)

def test_parallel():
    def reducer(one, two, three):
        return f"{one}: One | {two}: Two | {three}: Three"

    join_target = parallel(
        targets=[
            dummy(template="One", key="one"),
            dummy(template="Two", key="two"),
            dummy(template="Tree", key="three")
        ],
        reducer=reducer
    )

    result = join_target()
    assert result == "One: One | Two: Two | Tree: Three", \
        f"Expected 'One: One | Two: Two | Tree: Three', got {result}"
