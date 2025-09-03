from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from trustcall import create_extractor

from auxiliary import safe_lambda, accepted_keys, llm_test
from operators.agent import Agent, simple_agent

def extract(template, agent: Agent, schema: type[BaseModel], key: str = None):
    class Extract(Agent):
        def __init__(self):
            super().__init__(key=key)
            self.template = template
            self.accepted_keys = accepted_keys(template)
            self.schema = schema
            self.agent = agent

        def __call__(self, *args, **kwargs):
            result = agent(*args, **kwargs)

            prompt = self.template
            if callable(self.template):
                kwargs[self.agent.key] = result
                prompt = safe_lambda(self.template, self.accepted_keys, **kwargs)

            llm = ChatOpenAI(model="gpt-4o")
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

    llm_test(extract("Extract a bool value",
                     agent=simple_agent("Boolean: true"), schema=Extract))
