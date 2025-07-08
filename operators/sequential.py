from auxiliary import async_llm_test
from operators.dummy import dummy
from operators.infer import infer
from operators.target import Target

def sequential(targets: list[Target], reducer, key: str = None):
    class Sequential(Target):
        def __init__(self):
            super().__init__(key=key)
            self.targets = targets
            self.reducer = reducer
            assert len(self.targets) > 0, "Sequential operator requires at least one target."

        def __call__(self, *args, **kwargs):
            results = {}

            for target in self.targets:
                result = target(*args, **kwargs)
                kwargs[target.key] = target(*args, **kwargs)
                results[target.key] = result

            return self.reducer(**results)

    return Sequential()

def test_sequential():
    def reducer(one, two, three):
        return f"Results: {one}, {two}, {three}"

    seq = sequential(targets=[
        dummy("One", key="one"),
        dummy("Two", key="two"),
        dummy("Three", key="three")
    ], reducer=reducer)

    print(seq())

def test_sequential_2():
    inf = infer(lambda one, two: f"Combine these two stories: {one} and {two}")

    def reducer(one, two):
        return inf(one=one, two=two)

    seq = sequential(targets=[
        dummy("Cat", key="one"),
        dummy("Tree", key="two")], reducer=reducer)

    async_llm_test(seq)
