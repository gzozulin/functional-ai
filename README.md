# Functional AI Library: Deep Research Agent for Codebases

A modular AI orchestration library that lets you build LLM-powered pipelines using functional building blocks. It supports prompt composition, caching, looped refinement, tool integration, and testable components — all written in plain Python.

Whether you're generating UML diagrams, refining code context, or answering user queries, the architecture helps you reason about agent logic clearly and modularly. While it’s designed with source code research in mind, it can be applied to any multi-step LLM workflow.

## Features

- ✅ Composable Targets – Every unit of logic is a callable class with clearly defined inputs and outputs.
- 🔁 Looping, Caching, Sequencing – Functional-style orchestration with `infer`, `sequential`, `loop`, and `cache` blocks.
- 🧪 Isolated Testing – Each step is easy to test with predictable inputs, no side effects.
- 🛠 Tool Support – Built-in tools for file access, content paging, and external lookups.
- 📦 Backend-agnostic – Compatible with any LLM backend supporting agent-style execution.

# Example Usage

Using the library requires only a minimal setup. You define your pipeline using simple functional components, then invoke it with a request:

```python
await get_backend().create_session()

context_collector = fai.infer(
    context_collector_template, tools=[list_files, cat_file],
    key="context")

context_critic = fai.loopn(
    lambda context: f"Keep only code, lines & files: {wrap(context)}",
    fai.infer(context_critic_template, tools=[list_files, cat_file], key="context"),
    count=MAX_CTX_ITERATIONS,
    key="context")

context_full = fai.cache(
    fai.sequential(
        lambda context: f"Create a report and include all code:{wrap(context)}",
        targets=[context_collector, context_critic]),
    key="context")

uml_chart = fai.transform(uml_chart_template, context_full, key="uml")
pseudocode = fai.transform(pseudocode_template, context_full, key="pseudo")

user_reply = fai.parallel(
    user_reply_template, targets=[context_full, uml_chart, pseudocode],
    tools=[query_wiki])

print(user_reply(request="How is input handled in this system?"))
```

Before running any queries, get_backend().create_session() must be called to initialize the LLM session. This enables backend-specific features like tool access and streaming support.

Tools such are passed explicitly to each component when defined. This makes the behavior of the pipeline transparent and configurable — you know exactly which operations are available at each stage.

You can embed the agent pipeline inside larger apps (e.g., web UIs, background workers) or run it interactively in a CLI or notebook. The components are just Python callables — you can invoke them directly and inspect intermediate outputs.

If an error occurs — such as a missing file, invalid prompt, or backend failure — the system raises structured exceptions with clear messages, making it easy to trace and debug.

# Architecture Overview

At the core of the library is the concept of a Target: a minimal class that defines a callable unit of computation. Every component — from single prompt calls to multi-step chains — extends this base. It provides a consistent interface for execution while remaining lightweight and composable.

Each Target has a unique key, which determines where its result is stored in the pipeline’s shared state. When a component runs, its output is inserted into the kwargs dictionary under this key. This mechanism allows downstream targets to reference the output of upstream ones by name, keeping state passing explicit and traceable.

The library provides several functional combinators to build agents out of smaller pieces:

* infer(template) – single prompt call, optionally with tools
* sequential(template, [target1, target2, ...]) – runs each target in order and builds a final prompt
* loop(template, target, condition) – repeats execution while a condition holds
* loopn(template, target, count) – fixed iteration version of loop
* cache(target) – memoizes a target based on input arguments
* join(template, [targets], tools) – merges multiple sources into a final synthesis step

State is passed between all these components through keyword arguments (**kwargs). Each step reads what it needs and adds its own result, keeping everything side-effect-free and easy to follow.

Rather than relying on a graph or node-based model, the architecture favors flat, functional composition. This choice makes it easier to reason about flow, debug intermediate outputs, and write unit tests. Since targets are pure functions, they can be reused freely across pipelines without worrying about shared state or lifecycle side effects.

The result is a system where each block does one thing well, and all parts — from prompt execution to synthesis — are modular, testable, and extensible.

## Testing

The library is built with testability in mind. Every component — whether it's a prompt wrapper, sequential block, or loop — is implemented as a standalone `Target`. This makes it easy to test each part individually without needing to run the full pipeline.

To simplify test execution, the utility `async_llm_test()` is provided. It handles session setup and asynchronous invocation, allowing you to run and inspect any `Target` with a single call:

```python
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
```
While LLM responses are inherently non-deterministic, the structure of prompts and the use of tool inputs help keep results stable enough for practical testing. Most steps are either cached or templated tightly enough to make verification consistent across runs.

The use of key-based state passing ensures that tests can remain isolated. Each component reads and writes only its assigned variables, which avoids global state mutations and makes it easier to assert correctness at each step.

Caching also plays a role in test speed and reproducibility. Once a component has been run with a given set of arguments, it can be reused in later tests without recomputing, which is especially useful during development or when integrating with expensive backends.

## Backend Support

A backend in this library provides the runtime environment for LLM execution. It is responsible for setting up sessions, managing tool access, configuring the model, and executing prompts.

Before using any agent, you must call:

```python
await get_backend().create_session()
```

This ensures that the session context is initialized — typically including user ID, app metadata, and authentication with the LLM provider.

The default backend is GoogleAdkBackend, which wraps a ReAct-style agent using the Google Assistant Developer Kit (ADK). It supports streaming output, tool usage, and agent configuration (e.g., instructions, description, tool restrictions).

However, the system is backend-agnostic by design. You can support OpenAI, Anthropic, or even local LLMs by implementing your own backend class and swapping it in:

```python
def get_backend():
    return MyCustomBackend()
```

As long as the backend exposes create_runner() and call_agent(), it can be plugged into the rest of the system without modifying any functional components. This makes it easy to port the entire agent pipeline across platforms or environments.

## Planned Features

The library is stable and functional, but several improvements are planned to enhance usability, debugging, and flexibility:

- [ ] **Progress Reporting** – Display intermediate steps and prompt stages during execution.
- [ ] **Debug Mode** – Optional verbose output for tools, templates, and LLM calls.
- [ ] **Async Tool Support** – Allow tools to be defined as `async def`, enabling concurrent I/O or network-bound operations.
- [ ] **Dynamic Prompt Refinement** – Adjust prompts on the fly based on LLM feedback or runtime observations.
- [ ] **Web & API Interfaces** – Run the agent as a web service or integrate into an existing API layer.
- [ ] **Better Error Reporting** – More granular exceptions with structured logs for backend and tool failures.
- [ ] **Template Injection & Modularity** – Externalize prompt templates for easier editing, swapping, or multilingual support.

# Contributions Welcome
Community contributions are welcome — especially around tool extensions, backend integrations, and improving developer ergonomics.
Pull requests, feature suggestions, and experiments are all encouraged. If you're building your own research or agent pipelines, feel free to fork and adapt.
