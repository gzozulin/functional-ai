import asyncio
import inspect

from dotenv import load_dotenv

from backends.google_adk import get_backend

def accepted_keys(func):
    return set(inspect.signature(func).parameters.keys()) if callable(func) else set()

def safe_lambda(lmbda, keys, *args, **kwargs):
    accepted_args = {k: v for k, v in kwargs.items() if k in keys}
    missing_args = keys - set(accepted_args.keys())
    missing_args = {k: None for k in missing_args}
    all_args = accepted_args | missing_args
    return lmbda(*args, **all_args)

def async_llm_test(call, *args, **kwargs):
    load_dotenv()

    async def wrapper():
        await get_backend().create_session()
        print(call(*args, **kwargs))

    asyncio.run(wrapper())

def print_success_green(text):
    print(f"\033[92m{text}\033[0m")

def print_error_red(text):
    print(f"\033[91m{text}\033[0m")

def print_debug_yellow(text):
    print(f"\033[93m{text}\033[0m")

def print_llm_blue(text):
    print(f"\033[94m{text}\033[0m")

def print_user_default(text):
    print(text)

def print_dash(char: str = '-', count: int = 80):
    print(char * count)
