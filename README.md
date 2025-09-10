
<!-- Banner / Title -->
<div align="center">
  <img src="static/kandinsky.jpg" width="490" height="343" alt="DeepMCPAgent Logo"/>

  <h1>🤖 Functional AI</h1>
  <p><strong>A backend-agnostic, modular AI orchestration framework for building LLM-powered pipelines with functional building blocks.</strong></p>

  <!-- Badges -->
  <p>
    <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg"></a>
    <a href="#"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-beta-orange.svg"></a>
  </p>
</div>

## Features

- 🧩 Composable Agents – Every unit of logic is a callable class with clearly defined inputs and outputs
- 🔁 Looping, Caching, Sequencing – Functional-style orchestration with built-in blocks
- 🎯 Guardrails and Validation – Input/output validation, retry logic, and error handling
- 🚀 Parallel Execution – Run independent steps concurrently to speed up pipelines
- 🧪 Isolated Testing – Each step is easy to test with predictable inputs, no side effects
- 🛠 Tool Support – Built-in support for tools and MCP extensions with no boilerplate
- 📦 Backend-agnostic – Compatible with any LLM backend supporting agent-style execution

## Example Usage

[Using the library](examples/fai_chat.py) requires only a minimal setup. You define your pipeline using simple functional components, then invoke it with a request:

```python
import operators as fai

def cat_file(file_path: str, page: int) -> dict:
    ...

def query_wiki(query: str) -> dict:
    ...

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
            tools=[cat_file, query_wiki],
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

if __name__ == '__main__':
    print_success_green(f"German: {fai_german()}")
    print_success_green(f"Chinese: {fai_chinese()}")
    print_success_green(f"Russian: {fai_russian()}")
```

## Architecture Overview

At the core of the library is the concept of an Agent: a minimal **ReAct-style component** that defines a callable unit of computation. Every block — from single prompt calls to multi-step chains — extends this base. It provides a consistent interface for execution while remaining lightweight and composable.

**Each Agent has a unique key**, which determines where its result is stored in the pipeline’s shared state. When a component runs, its output is inserted into the kwargs dictionary under this key. This mechanism allows downstream targets to reference the output of upstream ones by name, keeping state passing explicit and traceable.

State is passed between all these components through keyword arguments (**kwargs). Each step reads what it needs and adds its own result, **keeping everything side-effect-free and easy to follow**.

Rather than relying on a graph or node-based model, **the architecture favors flat, functional composition.** This choice makes it easier to reason about flow, debug intermediate outputs, and write unit tests. Since targets are pure functions, **they can be reused freely** across pipelines without worrying about shared state or lifecycle side effects.

The result is a system where e**ach block does one thing well**, and all parts — from prompt execution to synthesis — **are modular, testable, and extensible**.

## Backend Support

A backend in this library provides the runtime environment for LLM execution. It is responsible for setting up sessions, managing tool access, configuring the model, and executing prompts.

The default backend is GoogleAdkBackend, which wraps a ReAct-style agent using the Google Assistant Developer Kit (ADK). It supports streaming output, tool usage, and agent configuration (e.g., instructions, description, tool restrictions).

However, the system is backend-agnostic by design. You can support OpenAI, Anthropic, or even local LLMs by implementing your own backend class and swapping it in:

```python
def get_backend():
    return MyCustomBackend()
```

As long as the backend exposes create_runner() and call_agent(), it can be plugged into the rest of the system without modifying any functional components. This makes it easy to port the entire agent pipeline across platforms or environments.

## Contributions Welcome
Community contributions are welcome — especially around tool extensions, backend integrations, and improving developer ergonomics.
Pull requests, feature suggestions, and experiments are all encouraged. If you're building your own research or agent pipelines, feel free to fork and adapt.
