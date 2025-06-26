# Functional AI Library Dev Journal: Implementing Deep Research Agent for Doom Codebase
Every software library is only as good as projects that it powers. Building functional components is one thing. Putting them to work in a real, LLM-powered agent is another. This article walks through the development of a modular AI research agent built to analyze and explain the Doom codebase. Rather than wrapping a monolithic prompt in a script, we break down the process into composable, testable blocks — each implemented as a pure Python callable.

From context collection and iterative refinement to UML generation and synthesis, every step in the pipeline is structured for clarity, reuse, and controlled execution. Under the hood, the agent uses a ReAct-style backend, external tools like Wikipedia and filesystem inspection, and a caching mechanism to stay efficient.

Whether you’re building your own AI-assisted research stack or just want to see how functionally composed LLM workflows operate in practice, this journal aims to demystify the architecture, one layer at a time.

The library is available on GitHub at https://github.com/gzozulin/functional-ai

Here is what we will cover in this article:
* A Quick Primer on Functional Components
* Prompt Engineering for Deep Research
* Agentic Tools and Their Role
* Deep Research Agent Implementation
* GoogleADK Backend Setup
* Testing and Running the Agent

## A Quick Primer on Functional Components

In this architecture, the Target class defines a minimal contract for any executable unit in the research system. It provides a standard interface via __call__() and introduces a key-based naming convention, which helps with result tracking and passing intermediate values between steps. Each Target subclass can use the key to insert its output into the shared kwargs, ensuring that downstream components receive the correct data. The LlmTarget extends this base by binding to an LLM agent and runner, allowing it to send templated prompts for language model processing. This distinction is essential: while a plain Target might perform a fixed computation, LlmTarget connects abstract tasks to the LLM backend, enabling dynamic inference.

```python
class Target:
    def __init__(self, key: str = None):
        self._key = key

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden in subclasses.")

    @property
    def key(self):
        return self._key if self._key is not None else 'it'

class LlmTarget(Target):
    def __init__(self, llm: str = None, tools=None, schema=None, key: str = None):
        super().__init__(key=key)

        if llm is None:
            llm = MODEL_GPT_4O_MINI  # Default to GPT-4O Mini

        if tools is None:
            tools = []

        self.agent, self.runner = get_backend().create_runner(llm, tools, schema)
```

The three higher-order functions — infer, sequential, and loop/loopn — represent different orchestration strategies for chaining or repeating LLM-powered tasks. 

The infer() function wraps a single prompt template into a callable Target. It’s designed for simple inference: given a request and optional tools, it produces a single LLM query. If the template is a function, it gets executed with filtered arguments using safe_lambda() to avoid passing unexpected parameters. This keeps the logic clean and minimizes prompt errors.

```python
def infer(template, llm: str = None, tools: list = None, key: str = None):
    class Infer(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools, key=key)
            self.template = template
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            prompt = self.template

            if callable(self.template):
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Infer()
```

The sequential() function composes a chain of targets. Each one is executed in order, and its result is inserted into the shared argument dictionary under its key. The final prompt is built from all accumulated data using the provided template. This is ideal for step-by-step workflows where each stage depends on the outputs of previous ones.

```python
def sequential(template, targets: list[Target], llm: str = None, tools: list = None, key: str = None):
    class Loop(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools, key=key)
            self.template = template
            self.targets = targets
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            for target in self.targets:
                kwargs[target.key] = target(*args, **kwargs)

            prompt = self.template
            if callable(self.template):
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Loop()
```

The loop() and its fixed-repeat variant loopn() repeatedly call a single target while a condition is true. Each iteration receives the index as idx and the previous result via the target's key. Once the loop ends, a final prompt is built. This construct is useful for refinement cycles, retry loops, or any task that benefits from incremental passes.

```python
def loop(template, target: Target, condition, llm: str = None, tools: list = None, key: str = None):
    class Loop(LlmTarget):
        def __init__(self):
            super().__init__(llm=llm, tools=tools, key=key)
            self.target = target
            self.template = template
            self.condition = condition
            self.accepted_keys = set(inspect.signature(template).parameters.keys()) \
                if callable(self.template) else set()

        def __call__(self, *args, **kwargs):
            index = 0
            while condition(idx=index):
                kwargs['idx'] = index
                kwargs[target.key] = self.target(*args, **kwargs)
                index += 1

            prompt = self.template
            if callable(self.template):
                prompt = safe_lambda(self.template, self.accepted_keys, *args, **kwargs)

            return get_backend().call_agent(prompt, self.runner)

    return Loop()

def loopn(template, target: Target, count, llm: str = None, tools: list = None, key: str = None):
    return loop(template, target, lambda idx: idx < count, llm, tools, key)
```

To avoid redundant calls, the cache() wrapper provides a mechanism to reuse results as long as inputs remain unchanged. It tracks a hash of arguments to determine when recomputation is necessary.

```python
def cache(target: Target, key: str = None) -> Target:
    class Cache(Target):
        def __init__(self):
            super().__init__(key=key)
            self.cache = None
            self.cache_key = None

        def _update_cache_key(self, **kwargs):
            cache_key = hash(tuple(sorted(kwargs.items())))
            if self.cache_key is None or self.cache_key != cache_key:
                self.cache = None
                self.cache_key = cache_key

        def __call__(self, *args, **kwargs):
            self._update_cache_key(**kwargs)

            if self.cache is not None:
                return self.cache

            self.cache = target(*args, **kwargs)
            return self.cache

    return Cache()
```

Finally, these components are fully composable: a loop can wrap a sequential, which can contain cached infer steps. This design encourages a declarative, functionally pure construction of research pipelines, where state is passed explicitly, and each computation remains testable in isolation.

## Prompt Engineering for Deep Research

At the heart of this system lies a set of focused prompt templates, each guiding the agent through a specific stage of the research process — from raw code extraction to user-facing explanation.

Each of the five prompt templates serves a distinct role in the deep research workflow. The context_collector_template gathers raw code, the context_critic_template refines that collection, while uml_chart_template and pseudocode_template translate the code into visual and logical formats. The final user_reply_template ties everything together into a comprehensive response. Together, these templates form a fixed, modular pipeline tailored for software codebase research.

The context_collector_template is intentionally strict: it instructs the agent not to explain or modify code. This design prevents hallucination or unwanted interpretation by the LLM. By limiting the agent to collection only, the prompt ensures that data passed downstream is grounded in source material and not speculative.

```python
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
```

The critic phase complements this by improving the precision of collected code. The context_critic_template is structured to help the agent act like an editor — identifying missing files, removing unrelated lines, and suggesting more relevant segments. This two-pass strategy results in both broader coverage and higher output quality.

```python
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
```

The uml_chart_template and pseudocode_template are designed to convert the refined code into more abstract representations.

```python
def uml_chart_template(request, context):
    return (f"Generate a UML chart in ASCII based on the request: {request}\n"
            f"And the context: {wrap(context)}"
            f"Reply only with the UML chart in text format.")

def pseudocode_template(request, context):
    return (f"Generate pseudocode based on the request: {request}\n"
            f"And the context: {wrap(context)}"
            f"Reply only with the pseudocode in text format.")
```

Longer term, these prompt components could support chaining or refinement based on the agent’s performance. For example, critic feedback could adjust the collector’s query or iterate on pseudocode generation. While currently static, the format is future-proofed for such extensions.

```python
def user_reply_template(request, context, uml, pseudo):
    return (f"Based on the request: {wrap(request)}"
            f"And the collected context: {wrap(context)}"
            f"And the UML chart: {wrap(uml)}"
            f"And the pseudocode: {wrap(pseudo)}"
            f"Provide a comprehensive reply to the user request:{wrap(request)}"
            f"Use Wikipedia to find additional complimentary information.")
```

Finally, each prompt is just a regular Python callable. This makes the system easy to use and debug. Developers can pass arguments directly, print the result, or test prompt formatting without any agent execution. This low-friction interface encourages experimentation and reduces coupling between logic and LLM behavior.

## Agentic Tools and Their Role

This chapter defines a few basic agent tools that the LLM can call during execution. While the exact definitions are straightforward — and likely familiar to any experienced developer — they demonstrate the typical structure and error handling conventions used throughout the system. If you've already implemented tools for LangChain or similar agent frameworks, you can likely skim or skip this section.

Each tool plays a specific role in grounding the agent’s actions in real data. list_files lets the agent explore directory contents, cat_file allows incremental access to source files, and query_wiki provides external context where needed. These tools are not always used — they're selectively attached to agent instances that require file access or external lookup during prompt execution.

```python
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
```

The cat_file tool implements paging, reading files in 250-line chunks. This avoids overwhelming the agent or consuming too much context space in large files. It also gives the agent a way to ask for the “next page”.

```python
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
```

We also define a query_wiki tool that allows the agent to look up terms on Wikipedia. This is useful for providing additional context or definitions that may not be present in the codebase itself.

```python
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
```

All tools are currently written as synchronous functions for simplicity. However, in a real backend — especially one involving remote APIs or I/O-bound operations — async variants would be preferable. Wrapping these in coroutine-safe runners or using async def is a straightforward future upgrade.

The functional approach and the immutable nature of the state allows for simple and effective asynchronous calls. Each tool can be executed independently, and their results can be passed to the agent without side effects. This design ensures that the agent's state remains consistent and predictable, which is crucial for complex operations.

## Deep Research Agent Implementation

This section assembles all the previously defined components into a working deep research pipeline. Given a user request — such as “How is the BSP subsystem implemented?” — the system executes a structured sequence of steps: it collects raw code, refines that collection, and generates multiple forms of representation (UML, pseudocode), before synthesizing a final reply.

The structure follows a modular design composed of infer, loopn, sequential, cache, transform, and join. First, context_collector retrieves potentially relevant code using filesystem tools. Then, context_critic is invoked in a loop to iteratively refine that result. These two are composed using sequential, and wrapped in a cache() call to avoid redundant execution if the same request is repeated.

```python
context_collector = fai.infer(
    context_collector_template, tools=[list_files, cat_file],
    key="context")
```

The context_critic component uses loopn — a fixed-count loop — allowing multiple refinement passes over the collected code. This is a lightweight but effective way to improve context quality without introducing uncontrolled recursion or retry logic. The number of iterations (MAX_CTX_ITERATIONS) can be tuned to balance cost and completeness.

```python
context_critic = fai.loopn(
    lambda context: f"Keep only code, lines & files: {wrap(context)}",
        fai.infer(context_critic_template, tools=[list_files, cat_file], key="context"),
        count=MAX_CTX_ITERATIONS,
    key="context")
```

Downstream, context_full is used as the input to two transform steps: one for UML chart generation, and another for pseudocode. Keeping these operations separate simplifies debugging and allows for independent modification of templates or output formatting. It also makes it easier to test or swap either module without affecting the rest of the pipeline.

```python
context_full = fai.cache(
    fai.sequential(
        lambda context: f"Create a report and include all code:{wrap(context)}",
        targets=[context_collector, context_critic]),
    key="context")

uml_chart = fai.transform(uml_chart_template, context_full, key="uml")
pseudocode = fai.transform(pseudocode_template, context_full, key="pseudo")
```

The user_reply component is where everything comes together. It receives the final context, UML diagram, and pseudocode, and synthesizes a comprehensive answer to the user's request. This step doesn’t merely combine text — it also invokes query_wiki, bringing in external knowledge when necessary to enrich the reply.

```python
user_reply = fai.join(
    user_reply_template, targets=[context_full, uml_chart, pseudocode],
    tools=[query_wiki])
```

From a testing and reliability standpoint, the design supports immutability and key-based state passing. Each component produces a deterministic output based only on its input arguments. This makes the pipeline easy to test in isolation — for example, a developer can call context_full() or uml_chart() with a fake context and verify results without needing an actual LLM session.

The modularity of this setup allows reuse and reconfiguration. For instance, context_full could be plugged into a bug analysis agent or documentation generator without change. Templates, tools, and targets can be swapped freely as long as they adhere to the same callable interface and use consistent keys.

## GoogleADK Backend Setup

The GoogleAdkBackend class acts as the core integration layer between functional components and the LLM infrastructure. It encapsulates the session setup, agent construction, and low-level execution logic. From the outside, everything built with Target just calls get_backend().call_agent(...), so this class is the actual runtime behind the functional interface.

The method create_runner() returns both the agent and a runner. The agent encapsulates logic, tools, and metadata like instructions and schemas. The runner manages execution — session, user context, and streaming output. Keeping them separate allows more flexibility: multiple runners could share one agent, and the agent can be serialized or reused independently of any live session.

The model setup is handled through LiteLlm(model=llm), a wrapper over the actual LLM backend. In this case, it’s designed to work with the GoogleADK (Assistant Developer Kit) interface, but it can also be adapted to OpenAI, Anthropic, or local models. The abstraction allows swapping the underlying model without changing the functional code.

```python
class GoogleAdkBackend:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.session = None

    async def create_session(self):
        self.session = await self.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    def create_runner(self, llm: str = None, tools=None, schema=None):
        target_agent = LlmAgent(
            model=LiteLlm(model=llm),
            name="functional_ai_agent",
            instruction="You are a helpful assistant",
            description="An agent that performs tasks based on instructions",
            output_schema=schema,
            output_key="result",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
            tools=tools)

        runner = Runner(
            agent=target_agent,
            app_name=APP_NAME,
            session_service=self.session_service)

        return target_agent, runner

    @staticmethod
    def call_agent(query: str, runner) -> str:
        content = types.Content(role='user', parts=[types.Part(text=query)])
        for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    return event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    raise RuntimeError(f"Agent escalated: {event.error_message or 'No specific message.'}")
                break

        raise RuntimeError("No final response received from the agent.")
```

The call_agent() method executes a full ReAct-style agent. It builds a structured message from the user prompt, sends it through the runner, and waits for a final response. The loop processes streamed agent events — tool calls, intermediate steps, and decisions — until a final textual output is returned. This makes the system compatible with long-running, tool-augmented agents that make decisions in steps.

Error handling is also covered. If no final response is emitted, or if the agent escalates due to missing tools or unsupported actions, the method raises an exception. This guards against silent failures and makes it easier to debug tool integration issues or prompt misalignment.

Importantly, this backend is modular. It can be swapped out with another LLM provider by replacing the create_runner() and call_agent() methods, as long as the interface stays compatible. This keeps the rest of the framework backend-agnostic — targets, prompts, and the pipeline structure don’t need to change when switching from GoogleADK to another LangGraph for example.

## Testing and Running the Agent

Testing each component in isolation is a core design goal of this system. Since each target is a pure callable — often wrapping a single prompt — it can be invoked with fixed arguments and expected to return a consistent, meaningful output. This makes the pipeline easy to verify incrementally and significantly simplifies debugging, especially when a step fails to return what downstream prompts expect.

The test functions use async_llm_test(), a utility that wraps any target and runs it in an async environment. This is necessary because some backends (like GoogleADK) require session setup via coroutines. Under the hood, this function prepares the environment, runs the prompt, and prints the output — making it suitable for both development checks and reproducible testing.

One challenge in testing LLM-driven agents is output variation. Since the model may return slightly different text on each run, tests are not strictly deterministic. However, most prompts are structured enough — and tool usage precise enough — that the functional outputs remain stable. For consistency, shared context like kwargs is passed explicitly between components, ensuring reproducible behavior across tests.

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

The interactive loop in the final run_agent() block allows the system to run as a simple command-line research assistant. It reads user input, runs the full pipeline, and prints the final synthesized reply. While currently designed as a CLI, the same logic could be reused in a web app, background worker, or API server with minimal changes.

Planned enhancements include adding progress indicators and debug configurations. For example, intermediate results like raw collected code, refined context, or intermediate tool calls could be optionally printed or logged. This would make it easier to trace reasoning, validate agent steps, and support transparent inspection during development or production usage.

```python
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
```

And finally, this chapter would not be complete without an actual example of interacting with the agent. Here are a few trivial questions from me with replies from the agent:
<example>

And that is all for this article, folks!

***

I hope you liked the article as much as I did writing the code for it. If you need any help with code or projects or have any questions, feel free to reach out. I would be happy to help. If you need the full version of the code with all the imports, please ping me via DM or in the comments. I also post on X @ https://x.com/gzozulin to update periodically about the project and share new ideas.
