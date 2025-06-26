import inspect

from auxiliary import safe_lambda
from backends.google_adk import get_backend
from operators.target import Target, LlmTarget

def loop(template, target: Target, condition, llm: str = None, tools: list = None, key: str = None):
    class Loop(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools, key=key)
            self.target = target
            self.template = template
            self.condition = condition
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            index = 0
            while condition(idx=index):
                kwargs['idx'] = index
                kwargs[target.key] = self.target(*args, **kwargs)
                index += 1

            prompt = self.template
            if callable(self.template):
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Loop()

def loopn(template, target: Target, count, llm: str = None, tools: list = None, key: str = None):
    return loop(template, target, lambda idx: idx < count, llm, tools, key)
