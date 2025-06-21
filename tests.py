# Core Ideas:
# 0. Keep It Simple Stupid (KISS).
# 1. Functional programming style with decorators.
# 2. Stick to normal Python syntax.

import asyncio

from dotenv import load_dotenv

import operators as fai
from backends.google_adk import get_backend

load_dotenv()

transform = (
    fai.transform(lambda trg, mul: f"Multiply {trg} by {mul}",
        fai.infer(lambda left, right: f"Calculate {left} + {right}")))

join = fai.join(lambda trg: f"Multiply {trg[0]} by {trg[1]}",
    [
        fai.infer(lambda left, right: f"Calculate {left} + {right}"),
        fai.infer(lambda left, right: f"Calculate {left} - {right}"),
    ]
)

loop = fai.loop(lambda trg: f"Combine sentences into a list: {[t for t in trg]}",
    fai.infer(lambda idx, prev: f"Create a sentence about a child who is {idx} years old and "
                                f"has a pet of {prev[idx] if idx < len(prev) else 'none'} years."),
    condition=lambda idx, prev: idx < 3
)

if __name__ == '__main__':
    async def run_agent():
        await get_backend().create_session()
        #print(calculator(left=5, right=10, mul=3))
        #print(join(left=5, right=10))
        print(loop())

    asyncio.run(run_agent())

