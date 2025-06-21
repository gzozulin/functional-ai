import inspect

from auxiliary import safe_lambda
from backends.google_adk import get_backend
from operators.target import Target, LlmTarget

def loop(template, target: Target, condition, llm: str = None, tools: list = None):
    class Loop(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools)
            self.target = target
            self.template = template
            self.condition = condition
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            index = 0
            results = []
            while condition(idx=index, prev=results):
                kwargs['idx'] = index
                kwargs['prev'] = results
                results.append(self.target(*args, **kwargs))
                index += 1

            prompt = self.template
            if callable(self.template):
                kwargs['trg'] = results
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Loop()
