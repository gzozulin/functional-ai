from concurrent.futures import ThreadPoolExecutor

from operators.agent import Agent, simple_agent

def fork(agent: Agent, mapper, reducer, key: str = None) -> Agent:
    class Fork(Agent):
        def __init__(self):
            super().__init__(key=key)
            self._agent = agent
            self._mapper = mapper
            self._reducer = reducer

        def __call__(self, *args, **kwargs):
            result = {self._agent.key: self._agent(*args, **kwargs)}
            agents = mapper(**result)

            with ThreadPoolExecutor() as executor:
                results = [executor.submit(trg, *args, **kwargs) for trg in agents]
                results = [f.result() for f in results]

            return self._reducer(results)

    return Fork()

def test_fork():
    def example_mapper(dum):
        return [simple_agent(f"Mapped 1: {dum}"),
                simple_agent(f"Mapped 2: {dum}")]

    def example_reducer(mapped_results):
        return " | ".join(mapped_results)

    forked_agent = fork(
        agent=simple_agent("Hello, World!", key="dum"),
        mapper=example_mapper,
        reducer=example_reducer,
        key="forked_example"
    )

    assert forked_agent() == "Mapped 1: Hello, World! | Mapped 2: Hello, World!"
