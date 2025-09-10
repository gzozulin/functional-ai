import os
from typing import List

from wikipedia import wikipedia

import operators as fai
from auxiliary import print_success_green, print_llm_blue
from prompts import PromptBuilder

file_list_operators = os.listdir('../operators')
file_list_operators = [f for f in file_list_operators if not f.startswith('.') and not os.path.isdir(f)]
file_list_backends = os.listdir('../backends')
file_list_backends = [f for f in file_list_backends if not f.startswith('.') and not os.path.isdir(f)]

# ---------- Tool Functions ---------

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

    try:
        summary = wikipedia.summary(query, sentences=3)
        return {'status': 'success', 'output': summary}
    except Exception as e:
        return {'status': 'error', 'output': str(e)}

# --------- Agent ---------

def sub_agent_template(question: str, chat_history: List[str]) -> str:
    return (PromptBuilder()
            .file('static/agent_prompt').dash()
            .text("Available files in ../operators").tab()
            .text(', '.join(file_list_operators)).back().dash()
            .text("Available backends in ../backends").tab()
            .text(', '.join(file_list_backends)).back().dash()
            .chat(chat_history).dash()
            .text(f"Question: {question}").dash()
            .prompt)

def main_agent_template(chat_history: List[str],
                        file_agent: str, backend_agent: str, general_agent: str) -> str:
    return (PromptBuilder()
            .file('static/agent_prompt').dash()
            .text("Available files in ../operators").tab()
            .text(', '.join(file_list_operators)).back().dash()
            .text("Available backends in ../backends").tab()
            .text(', '.join(file_list_backends)).back().dash()
            .text(f"File agent response:\n{file_agent}").dash()
            .text(f"Backend agent response:\n{backend_agent}").dash()
            .text(f"General agent response:\n{general_agent}").dash()
            .chat(chat_history)
            .prompt)

def create_sub_agent(question: str, key: str) -> fai.Agent:
    return fai.retry(   # Retry on failure up to 3 times by default
        fai.ai_agent(   # Create a Re-Act style AI agent with tools
            template=lambda chat_history: sub_agent_template(question, chat_history),
            tools=[cat_file, query_wiki],  # Each subagent can have own tools
            key=key))

fai_agent = fai.ai_parallel(        # Run parallel research with multiple agents
    template=main_agent_template,   # Combine the results from subagents
    agents=[
        create_sub_agent("Find the file related to users' question", key="file_agent"),
        create_sub_agent("Find the backend related to users' question", key="backend_agent"),
        create_sub_agent("Find general information related to user's question", key="general_agent")
    ])

fai_chat = fai.cache(   # Cache and reuse the result of this chat
    fai.ai_summarize(   # Summarize the chat at the end
        fai.ai_chat(    # Chat with the user until they say '!done'
            agent=fai_agent, output_llm=print_llm_blue, input_user=input)))

fai_german = fai.ai_transform(  # Transform the cached result
    template=lambda it: f"Translate to German:\n\n{it}", agent=fai_chat)
fai_chinese = fai.ai_transform(
    template=lambda it: f"Translate to Chinese:\n\n{it}", agent=fai_chat)
fai_russian = fai.ai_transform(
    template=lambda it: f"Translate to Russian:\n\n{it}", agent=fai_chat)

# --------- Main ---------

if __name__ == '__main__':
    print_success_green(f"German: {fai_german()}")
    print_success_green(f"Chinese: {fai_chinese()}")
    print_success_green(f"Russian: {fai_russian()}")
