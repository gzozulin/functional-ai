# Core Ideas:
# 0. Keep It Simple Stupid (KISS).
# 1. Functional programming style with decorators.
# 2. Stick to normal Python syntax.

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import operators as fai
from backends.google_adk import get_backend

load_dotenv()

transform = (
    fai.transform(lambda it, mul: f"Multiply {it} by {mul}",
        fai.infer(lambda left, right: f"Calculate {left} + {right}")))

join = fai.join(lambda it: f"Multiply {it[0]} by {it[1]}",
    [
        fai.infer(lambda left, right: f"Calculate {left} + {right}"),
        fai.infer(lambda left, right: f"Calculate {left} - {right}"),
    ]
)

loop = fai.loop(lambda it: f"Combine sentences into a list: {[t for t in it]}",
    fai.infer(lambda idx, it: f"Create a sentence about a child who is {idx} years old and "
                              f"has a pet of {it[idx] if idx < len(it) else 'none'} years."),
    condition=lambda idx: idx < 3
)

class CapitalInfoOutput(BaseModel):
    capital: str = Field(description="The capital city of the country.")
    population_estimate: str = Field(description="An estimated population of the capital city.")

extract = fai.extract(lambda it: f"Extract information from the text: {it}",
                      target=fai.infer("Tell me how many people live in the capital city of France and what is its name."),
                      schema=CapitalInfoOutput)


def complete_story(knight: str, dragon: str) -> str:
    return f"Create a sentence about the {knight} and the {dragon} becoming friends"

sequential = fai.sequential(lambda it: f"Complete the story: {it}",
    [
        fai.infer("Create a sentence about a knight", key="knight"),
        fai.infer(lambda knight: f"Create a sentence about a dragon fighting the {knight}", key="dragon"),
        fai.infer(complete_story),
    ])

if __name__ == '__main__':
    async def run_agent():
        await get_backend().create_session()
        print(transform(left=5, right=10, mul=3))
        print(join(left=5, right=10))
        print(loop())
        print(extract())
        print(sequential())

    asyncio.run(run_agent())

