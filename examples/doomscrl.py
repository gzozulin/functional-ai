import asyncio
import os

from dotenv import load_dotenv
from wikipedia import wikipedia

import operators as fai
from auxiliary import async_llm_test

from backends.google_adk import get_backend

DOOM_DIR = '/home/greg/DOOM'
MAX_CTX_ITERATIONS = 2

load_dotenv()

# -------------------- Tools --------------------

def list_files(directory: str) -> dict:
    """List all files in the given directory.
    * Args:
        directory (str): The directory to list files from.
    * Returns:
        dict: A dictionary with status and output.
            - 'status': 'success' or 'error'
            - 'output': The list of files or an error message.
    """

    print(f'--- Tool: list_files called for {directory} ---')

    if not os.path.exists(directory):
        return {'status': 'error', 'output': f'Directory {directory} does not exist.'}

    command = f"ls -la {directory}"

    try:
        output = os.popen(command).read()
        return {'status': 'success', 'output': output}
    except Exception as e:
        return {'status': 'error', 'output': str(e)}

def cat_file(file_path: str, page: int) -> dict:
    """Read a file and return its content. Each page contains 250 lines or EOF.
    * Args:
        file_path (str): The path to the file to read.
        page (int): The page number to read (0-based).
    * Returns:
        dict: A dictionary with status and output.
            - 'status': 'success' or 'error'
            - 'output': The content of the file or an error message.
    """

    print(f'--- Tool: cat_file called for {file_path}, page {page} ---')

    if page < 0:
        return {'status': 'error', 'output': 'Page number must be non-negative.'}

    if not os.path.exists(file_path):
        return {'status': 'error', 'output': f'File {file_path} does not exist.'}

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start = page * 250
            end = start + 250

            if start >= len(lines):
                return {'status': 'error', 'output': 'End of file reached.'}

            content = ''.join(lines[start:end])
            return {'status': 'success', 'output': content}
    except Exception as e:
        return {'status': 'error', 'output': str(e)}

def query_wiki(query: str) -> dict:
    """Query Wikipedia for a given term.
    * Args:
        query (str): The term to search for on Wikipedia.
    * Returns:
        dict: A dictionary with status and output.
            - 'status': 'success' or 'error'
            - 'output': The summary of the Wikipedia page or an error message.
    """

    print(f'--- Tool: query_wiki called for {query} ---')

    try:
        summary = wikipedia.summary(query, sentences=3)
        return {'status': 'success', 'output': summary}
    except Exception as e:
        return {'status': 'error', 'output': str(e)}

# -------------- Prompts --------------------

def wrap(string: str) -> str:
    return f'\n{string}\n' + '-' * 80 + '\n'

def context_collector_template(request):
    return (
        "You are a code collector agent. Your only job is to collect raw code "
        "snippets from the Doom directory that are relevant to the user's request.\n\n"
        f"User request:\n"
        f"{wrap(request)}"
        
        f"The code is located in the DOOM_DIR: {DOOM_DIR}\n"
        
        "Do not explain or modify the code. Only collect raw snippets.\n"
        "Include file paths and line numbers if helpful.\n"
        "Consider any file type that might help answer the user's request, "
        "such as explaining behavior, generating UML, or writing pseudocode.\n\n"
        "Return only the relevant code snippets."
    )

def context_critic_template(request, context):
    return (
        "You are a critic agent. Your task is to review the collected code snippets "
        "and help improve their relevance to the user's request.\n\n"
        f"User request:\n"
        f"{wrap(request)}\n"
        
        f"Collected code snippets:\n"
        f"{wrap(context)}"
        
        "Carefully examine the collected code. Identify what should be added, "
        "removed, or replaced to better support the user's request.\n"
        
        "- Add missing files or parts related to request\n"
        "- Remove files or snippets because it's unrelated\n"
        "- Replace code with a more complete version\n\n"
        
        "Use tools and return the updated and improved code snippets.\n"
    )

def uml_chart_template(request, context):
    return (f"Generate a UML chart in ASCII based on the request: {request}\n"
            f"And the context: {wrap(context)}"
            f"Reply only with the UML chart in text format.")

def pseudocode_template(request, context):
    return (f"Generate pseudocode based on the request: {request}\n"
            f"And the context: {wrap(context)}"
            f"Reply only with the pseudocode in text format.")

def user_reply_template(request, context, uml, pseudo):
    return (f"Based on the request: {wrap(request)}"
            f"And the collected context: {wrap(context)}"
            f"And the UML chart: {wrap(uml)}"
            f"And the pseudocode: {wrap(pseudo)}"
            f"Provide a comprehensive reply to the user request:{wrap(request)}"
            f"Use Wikipedia to find additional complimentary information.")

# -------------- Agent --------------------

context_collector = fai.ai_agent(
    context_collector_template, tools=[list_files, cat_file],
    key="context")

context_critic = fai.loopn(
    lambda context: f"Keep only code, lines & files: {wrap(context)}",
        fai.ai_agent(context_critic_template, tools=[list_files, cat_file], key="context"),
        count=MAX_CTX_ITERATIONS,
    key="context")

context_full = fai.cache(
    fai.sequential(
        agents=[context_collector, context_critic]),
    key="context")

uml_chart = fai.transform(uml_chart_template, context_full, key="uml")
pseudocode = fai.transform(pseudocode_template, context_full, key="pseudo")

user_reply = fai.parallel(
    user_reply_template, agents=[context_full, uml_chart, pseudocode])

# -------------------- Tests --------------------

def test_context_collector():
    async_llm_test(context_collector, request="How the main loop works?")

def test_context_full():
    async_llm_test(context_full, request="How BSP subsystem works?")

def test_uml_chart():
    async_llm_test(uml_chart, request="What is the structure of the frame?")

def test_pseudocode():
    async_llm_test(pseudocode, request="How the networking implemented?")

def test_user_reply():
    async_llm_test(user_reply, request="How user's input is collected?")

if __name__ == '__main__':
    async def run_agent():
        await get_backend().create_session()

        while True:
            request = input("Ask me anything: ")
            if request.lower() in ['exit', 'quit']:
                print("Exiting the agent.")
                break

            print(user_reply(request=request))

    asyncio.run(run_agent())
