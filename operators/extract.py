from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from trustcall import create_extractor

from auxiliary import safe_lambda, accepted_keys, async_llm_test
from operators import dummy
from operators.target import Target

def extract(template, target: Target, schema: type[BaseModel], key: str = None):
    class Extract(Target):
        def __init__(self):
            super().__init__(key=key)
            self.template = template
            self.accepted_keys = accepted_keys(template)
            self.schema = schema
            self.target = target

        def __call__(self, *args, **kwargs):
            result = target(*args, **kwargs)

            prompt = self.template
            if callable(self.template):
                kwargs[self.target.key] = result
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            llm = ChatOpenAI(model="gpt-4o")  # Default to GPT-4O
            extractor = create_extractor(llm, tools=[self.schema])
            prompt_template = ChatPromptTemplate([('system', prompt)])
            result = extractor.invoke(prompt_template.format())
            return result["responses"][0]

    return Extract()

def test_extract():
    class Extract(BaseModel):
        boolean: bool

        def __repr__(self):
            return f"Extract(boolean={self.boolean})"

    async_llm_test(extract("Extract a bool value",
                           target=dummy("Boolean: true"), schema=Extract))
