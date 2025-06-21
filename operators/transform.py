import inspect

from auxiliary import safe_lambda
from backends.google_adk import get_backend
from operators.target import Target, LlmTarget

def transform(template, target: Target, llm: str = None, tools: list = None):
    class Transform(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools)
            self.template = template
            self.target = target
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            result = target(*args, **kwargs)

            prompt = self.template
            if callable(self.template):
                kwargs['trg'] = result
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Transform()
