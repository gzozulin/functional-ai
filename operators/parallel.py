from concurrent.futures import ThreadPoolExecutor

from auxiliary import safe_lambda, accepted_keys
from operators.agent import ai_agent
from operators.agent import Agent, simple_agent

def parallel(agents: list[Agent], reducer, key: str = None):
    class Parallel(Agent):
        def __init__(self):
            super().__init__(key=key)
            self.agents = agents
            self.reducer = reducer

            if reducer is not None:
                self.reducer_keys = accepted_keys(self.reducer)

        def __call__(self, *args, **kwargs):
            with ThreadPoolExecutor() as executor:
                results = [executor.submit(agent, *args, **kwargs) for agent in agents]
                results = [f.result() for f in results]
                results = {agent.key: result for agent, result in zip(self.agents, results)}

            if reducer is not None:
                return safe_lambda(self.reducer, self.reducer_keys, **results)
            else:
                return None

    return Parallel()

def ai_parallel(template, agents: list[Agent], llm: str = None, tools: list = None, key: str = None):
    def reducer(**kwargs):
        return ai_agent(template, llm, tools)(**kwargs)
    return parallel(agents, reducer, key)

def test_parallel():
    def reducer(one, two, three):
        return f"{one}: One | {two}: Two | {three}: Three"

    join_agent = parallel(
        agents=[
            simple_agent(call="One", key="one"),
            simple_agent(call="Two", key="two"),
            simple_agent(call="Tree", key="three")
        ],
        reducer=reducer
    )

    result = join_agent()
    assert result == "One: One | Two: Two | Tree: Three", \
        f"Expected 'One: One | Two: Two | Tree: Three', got {result}"
