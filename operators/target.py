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
    def __init__(self, llm: str = None, tools=None, schema=None, key: str = None):
        super().__init__(key=key)

        if llm is None:
            llm = MODEL_GPT_4O_MINI  # Default to GPT-4O Mini

        if tools is None:
            tools = []

        self.agent, self.runner = get_backend().create_runner(llm, tools, schema)
