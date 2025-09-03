from auxiliary import llm_test, accepted_keys, safe_lambda
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

def ai_summarize(agent: Agent, llm: str = None, tools: list = None, key: str = None):
    return ai_transform(template=lambda it: f"Summarize the following while keeping all the details:\n\n {it}",
                        agent=agent, llm=llm, tools=tools, key=key)

def test_transform():
    llm_test(transform(
        agent=simple_agent("Hard Work!"),
        transformer=lambda it: f"All done: {it}"
    ))

def test_transform_2():
    llm_test(ai_transform(
        agent=simple_agent("Hard Work!"),
        template=lambda it: f"What is {it}?",
    ))

def test_summarize():
    llm_test(ai_summarize(agent=simple_agent("Artificial Intelligence is the simulation "
                                    "of human intelligence processes by machines, "
                                    "especially computer systems. These processes "
                                    "include learning (the acquisition of information "
                                    "and rules for using the information), reasoning "
                                    "(using rules to reach approximate or definite conclusions), "
                                    "and self-correction. AI applications include "
                                    "expert systems, natural language processing (NLP), "
                                    "speech recognition, and machine vision.")))
