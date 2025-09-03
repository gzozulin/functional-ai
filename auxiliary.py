import asyncio
import inspect

from dotenv import load_dotenv

from backends.google_adk import get_backend

def accepted_keys(func):
    return set(inspect.signature(func).parameters.keys()) if callable(func) else set()

def safe_lambda(lmbda, keys, **kwargs):
    if 'kwargs' in keys:
        accepted_args = {k: v for k, v in kwargs.items()}  # all
    else:
        accepted_args = {k: v for k, v in kwargs.items() if k in keys}

    missing_args = keys - set(accepted_args.keys())
    missing_args = {k: None for k in missing_args}
    all_args = accepted_args | missing_args
    return lmbda(**all_args)

def llm_test(call, *args, **kwargs):
    load_dotenv()
    print(call(*args, **kwargs))

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

def template_history(history: list[str]) -> str:
    conversation_history = ""
    is_llm = False

    for entry in history:
        if is_llm:
            conversation_history += f">>> LLM:\n{entry}\n\n"
        else:
            conversation_history += f">>> User:\n{entry}\n\n"
        is_llm = not is_llm

    return conversation_history
