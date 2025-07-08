from dotenv import load_dotenv

from auxiliary import async_llm_test
from operators.target import LlmTarget

load_dotenv()

def infer(template, llm: str = None, tools: list = None, key: str = None):
    return LlmTarget(template=template, llm=llm, tools=tools, key=key)

def test_infer():
    def template_func(x):
        return f"Tell a short (2 sentences) story about {x}"

    async_llm_test(infer(template="Tell a short (2 sentences) story about a tree"))
    async_llm_test(infer(template=template_func), x="a cat")
