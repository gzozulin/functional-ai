from auxiliary import async_llm_test, accepted_keys, safe_lambda
from operators.agent import Agent, simple_agent, ai_agent

def transform(agent: Agent, transformer, key: str = None):
    class Transform(Agent):
        def __init__(self):
            super().__init__(key)
            self.agent = agent
            self.transformer = transformer

            if transformer is not None:
                self.transformer_keys = accepted_keys(self.transformer)

        def __call__(self, *args, **kwargs):
            kwargs[agent.key] = agent(*args, **kwargs)

            if self.transformer is not None:
                return safe_lambda(self.transformer, self.transformer_keys, **kwargs)
            else:
                return None

    return Transform()

def ai_transform(template, agent: Agent, llm: str = None, tools: list = None, key: str = None):
    def transformer(**kwargs):
        return ai_agent(template, llm, tools)(**kwargs)
    return transform(agent, transformer, key)

def test_transform():
    async_llm_test(transform(
        agent=simple_agent("Hard Work!"),
        transformer=lambda it: f"All done: {it}"
    ))

def test_transform_2():
    async_llm_test(ai_transform(
        agent=simple_agent("Hard Work!"),
        template=lambda it: f"What is {it}?",
    ))
