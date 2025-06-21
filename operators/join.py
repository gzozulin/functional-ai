import inspect

from auxiliary import safe_lambda
from backends.google_adk import get_backend
from operators.target import Target, LlmTarget

def join(template, targets: list[Target], llm: str = None, tools: list = None):
    class Join(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools)
            self.template = template
            self.targets = targets
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            results = [target(*args, **kwargs) for target in targets]  # todo: parallel

            prompt = self.template
            if callable(self.template):
                kwargs['trg'] = results
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Join()
