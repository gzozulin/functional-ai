from auxiliary import async_llm_test
from operators import infer
from operators.target import Target, LlmTarget

def transform(template, target: Target, llm: str = None, tools: list = None, key: str = None):
    class Transform(LlmTarget):
        def __init__(self):
            super().__init__(template=template, llm=llm, tools=tools, key=key)
            self.target = target

        def __call__(self, *args, **kwargs):
            kwargs[target.key] = target(*args, **kwargs)
            return super().__call__(*args, **kwargs)

    return Transform()

def test_transform():
    async_llm_test(transform(
        lambda it: f"Translate into German: {it}",
        target=infer("Tell a short (2 sentences) story about a cat")))
