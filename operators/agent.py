from dotenv import load_dotenv

from auxiliary import accepted_keys, safe_lambda
from auxiliary import llm_test
from backends.google_adk import get_backend, MODEL_GPT_4O

load_dotenv()

class Agent:
    def __init__(self, key: str = None):
        self._key = key

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden in subclasses.")

    @property
    def key(self):
        return self._key if self._key is not None else 'it'

class LlmAgent(Agent):
    def __init__(self, template, llm: str = None, tools=None, schema=None, key: str = None):
        super().__init__(key=key)
        self._template = template
        self._template_keys = accepted_keys(template)

        if llm is None:
            llm = MODEL_GPT_4O

        if tools is None:
            tools = []

        self.agent, self.runner, self.session = get_backend().create_runner(llm, tools, schema)

    def __call__(self, *args, **kwargs):
        prompt = self._template

        if callable(self._template):
            prompt = safe_lambda(self._template, self._template_keys, **kwargs)

        return get_backend().call_agent(prompt, self.runner, self.session)

def simple_agent(call, key=None):
    class SimpleAgent(Agent):
        def __init__(self):
            super().__init__(key=key)
            self.call = call
            self.call_keys = accepted_keys(call)

        def __call__(self, *args, **kwargs):
            if callable(self.call):
                return safe_lambda(self.call, self.call_keys, **kwargs)
            return call

    return SimpleAgent()

def ai_agent(template, llm: str = None, tools: list = None, key: str = None):
    return LlmAgent(template=template, llm=llm, tools=tools, key=key)

def test_ai_agent():
    def template_func(x):
        return f"Tell a short (2 sentences) story about {x}"

    llm_test(ai_agent(template="Tell a short (2 sentences) story about a tree"))
    llm_test(ai_agent(template=template_func), x="a cat")


def test_ai_agent_2():
    llm_test(ai_agent("Come up with a short 2 sentences story about a tree"))

