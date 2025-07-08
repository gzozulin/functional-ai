from auxiliary import accepted_keys, safe_lambda
from backends.google_adk import MODEL_GPT_4O_MINI, get_backend

class Target:
    def __init__(self, key: str = None):
        self._key = key

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden in subclasses.")

    @property
    def key(self):
        return self._key if self._key is not None else 'it'

class LlmTarget(Target):
    def __init__(self, template, llm: str = None, tools=None, schema=None, key: str = None):
        super().__init__(key=key)
        self._template = template
        self._template_keys = accepted_keys(template)

        if llm is None:
            llm = MODEL_GPT_4O_MINI  # Default to GPT-4O Mini

        if tools is None:
            tools = []

        self.agent, self.runner = get_backend().create_runner(llm, tools, schema)

    def __call__(self, *args, **kwargs):
        prompt = self._template

        if callable(self._template):
            prompt = safe_lambda(self._template, self._template_keys, *args, **kwargs)

        return get_backend().call_agent(prompt, self.runner)
