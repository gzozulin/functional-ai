import asyncio

from backends.google_adk import get_backend

def safe_lambda(lmbda, keys, *args, **kwargs):
    accepted_args = {k: v for k, v in kwargs.items() if k in keys}
    missing_args = keys - set(accepted_args.keys())
    missing_args = {k: None for k in missing_args}
    all_args = accepted_args | missing_args
    return lmbda(*args, **all_args)

def async_llm_test(call, *args, **kwargs):
    async def wrapper():
        await get_backend().create_session()
        print(call(*args, **kwargs))
    asyncio.run(wrapper())
