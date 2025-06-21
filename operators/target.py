from backends.google_adk import MODEL_GPT_4O_MINI, get_backend

class Target:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden in subclasses.")

class MultiTarget(Target):
    def __init__(self, targets: list[Target]):
        super().__init__()
        self.targets = targets

class LlmTarget(Target):
    def __init__(self, llm: str = None, tools=None):
        super().__init__()

        if llm is None:
            llm = MODEL_GPT_4O_MINI  # Default to GPT-4O Mini

        if tools is None:
            tools = []

        self.runner = get_backend().create_runner(llm, tools)


