from operators.agent import Agent, simple_agent

def loop(agent: Agent, condition, reducer, key: str = None):
    class Loop(Agent):
        def __init__(self):
            super().__init__(key=key)
            self.agent = agent
            self.condition = condition
            self.reducer = reducer

        def __call__(self, *args, **kwargs):
            results = []

            index = 0
            while condition(idx=index, **kwargs):
                result = self.agent(*args, **kwargs, idx=index)
                results.append(result)
                kwargs[agent.key] = result
                index += 1

            return self.reducer(results)

    return Loop()

def loopn(agent: Agent, count, key: str = None):
    return loop(agent, lambda idx: idx < count, key)

def test_loop():
    def agent_func(idx):
        return f"Iteration {idx}"

    def reducer(results):
        return " | ".join(results)

    loop_agent = loop(
        agent=simple_agent(call=agent_func),
        condition=lambda idx, **kwargs: idx < 5,
        reducer=reducer
    )

    result = loop_agent()
    assert result == "Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4", \
        "Loop should iterate 5 times and return the correct results"
