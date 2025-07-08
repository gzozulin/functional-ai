# Functional AI Library Dev Journal: Implementing AI Sci-Fi English Tutor

Okay, wanna see my secret weapon which really improved my English?

I decided to build an English tutor set in a Sci-Fi universe because that’s the kind of world I’d want to practice in. Language learning becomes more immersive when it’s wrapped in narrative — especially one filled with spaceships, alien politics, and advanced technologies. Instead of doing yet another fill-in-the-blank worksheet, I wanted each session to feel like an episode from a futuristic story I was living through.

Storytelling helps in ways traditional grammar drills don’t. It embeds language in a rich context, making new vocabulary and expressions more memorable. You’re not just learning passively; you’re navigating a universe, talking back to it, and getting feedback in return. And when you engage emotionally with a story, you're far more likely to retain what you learn.

For now, I’ve built it to run in the terminal — fast, distraction-free, and portable. There’s no need to fire up a browser or deal with animations; just type, read, and respond. This format also aligns with the idea of staying in “character” — like you're communicating with a futuristic onboard AI.

Each session is structured as a mini narrative loop: the system generates a story, selects a paragraph for you to read aloud, offers a rule to practice, gives you a task, and then evaluates your response. This design mirrors how we naturally learn — read, learn something new, try it, and get corrected.

The user profile is updated every session. When you respond to a task, the AI evaluates your answer and adjusts its idea of your skill level — not just with a score, but by influencing future rules and challenges. Over time, the agent becomes more attuned to your specific gaps.

The “practice rules” are not generic drills. They are short, focused objectives like “use past perfect tense correctly” or “avoid vague pronouns.” They're informed by your profile and the current context of the story, so you’re always applying what you learn instead of memorizing.

The agent is designed to be used repeatedly, evolving with you. Each run generates a new setting and new challenges, but your profile carries over. That makes it feel less like a test and more like an ongoing story-driven journey of improvement.

The library is available on GitHub at https://github.com/gzozulin/functional-ai

Here is what we will cover in this article:
* Map-Reduce With the Functional Flavor
* Creating a Universe in Parallel Threads
* Parallel Execution in Functional Programming
* Story, Rule, Task, and Examiner
* The AI Tutor: A Sci-Fi Adventure

## Map-Reduce With the Functional Flavor

To coordinate multiple AI calls efficiently, I introduced a functional fork() operator — a reusable construct that lets me fan out a single AI result into multiple branches, then bring their results back together with a reducer. This operator mirrors the map-reduce pattern but applies it to inference pipelines instead of raw data processing.

The motivation was straightforward: many steps in the AI tutor require breaking down a complex task (like universe generation) into smaller parts (like creating detailed reports for different aspects). Each part can be processed in parallel, and their outputs need to be merged later. That’s classic map-reduce, and adapting it into a functional style gave me composability, testability, and clarity.

By using fork(), I can define a starting point (target), a way to split it (mapper), and a way to merge the results (reducer). Each of these is passed in explicitly, keeping the logic decoupled and highly modular. The target runs first and its result is used to generate new Target instances via mapper. Those are run in parallel, and the final reducer brings the mapped outputs together into a single result.

```python
def fork(target: Target, mapper, reducer, key: str = None) -> Target:
    class Fork(Target):
        def __init__(self):
            super().__init__(key=key)
            self._target = target
            self._mapper = mapper
            self._reducer = reducer

        def __call__(self, *args, **kwargs):
            result = {self._target.key: self._target(*args, **kwargs)}
            targets = mapper(**result)

            with ThreadPoolExecutor() as executor:
                results = [executor.submit(trg, *args, **kwargs) for trg in targets]
                results = [f.result() for f in results]

            return self._reducer(results)

    return Fork()
```

To enable concurrent execution, I wrapped the mapped targets using Python’s ThreadPoolExecutor. It's a simple, effective choice for I/O-bound workloads like LLM inference. While other async patterns could work, the thread pool keeps it easy to reason about and integrates smoothly with existing blocking APIs.

```python
def test_fork():
    def example_mapper(dum):
        return [dummy(f"Mapped 1: {dum}"),
                dummy(f"Mapped 2: {dum}")]

    def example_reducer(mapped_results):
        return " | ".join(mapped_results)

    forked_target = fork(
        target=dummy("Hello, World!", key="dum"),
        mapper=example_mapper,
        reducer=example_reducer,
        key="forked_example"
    )

    assert forked_target() == "Mapped 1: Hello, World! | Mapped 2: Hello, World!"
```

The test case test_fork() demonstrates the pattern clearly: a dummy result is mapped into two strings, and the reducer joins them. This confirms both the correctness of the logic and that the mapper and reducer interact cleanly with the wrapped system.

## Creating a Universe in Parallel Threads

Rather than simply prompting the AI with “give me a story,” I wanted each tutoring session to be grounded in a unique and richly imagined Sci-Fi universe. The goal was to create something the user could return to and explore, session after session — a consistent world filled with rules, politics, tech, and mysteries. This deeper structure gives more weight to every task and makes the learning process more engaging.

To break down the creative process, I used three distinct prompt templates: one to generate aspects of the universe (like culture, technology, species), another to expand each aspect into a full report, and a final one to merge everything into a comprehensive universe description. This staged structure lets each step focus on a smaller, more manageable task — something LLMs handle better than giant prompts.

```python
def universe_details_template(it: str) -> str:
    return f"""
What are the key aspects of the Sci-Fi universe?
Examples: political systems, technology, culture, alien species, etc.
Answer with 10 aspects, each on a new line.
----------------------------------------------------------------------------
Universe:\n\n{it}
"""

def universe_detail_report_template(detail: str) -> str:
    return f"""
Create a detailed report on a described aspect of the Sci-Fi universe.
The report should be approximately 250 words long.
-----------------------------------------------------------------------------
Aspect:\n\n{detail}
"""

def universe_full_report_template(reports: list[str]) -> str:
    reports = "\n\n".join(reports)
    return f"""
Create a comprehensive report on the Sci-Fi universe.
The report should include all the details from the provided reports.
The report should be approximately 1000 words long.
-----------------------------------------------------------------------------
Reports:\n\n{reports}
"""
```

The function universe_details_mapper() takes the list of aspects and builds inference targets for each one. Each target is wrapped with fai.catch() to ensure robustness — if the first inference fails (e.g. because the LLM throttles or produces an error), it retries with a smaller fallback model like GPT-4o-mini. This is important because we're generating a chain of AI outputs, and a single failure shouldn't break the flow.

```python
universe = fai.cache(fai.infer(template="Create a setting for a Sci-Fi universe"))
universe_details = fai.transform(universe_details_template, target=universe)

def universe_details_mapper(it: str) -> list[Target]:
    details = it.splitlines()

    prompts = [universe_detail_report_template(detail)
               for detail in details if detail.strip()]

    return [fai.catch(
                target=fai.infer(template=prompt),
                exception=fai.infer(template=prompt, llm=MODEL_GPT_4O_MINI))
            for prompt in prompts]

def universe_full_reducer(it: list[str]) -> str:
    prompt = universe_full_report_template(it)
    return fai.infer(template=prompt)()

setting = fai.store(fai.fork(
    target=universe_details,
    mapper=universe_details_mapper,
    reducer=universe_full_reducer), key="sett", filename=".setting")
```

To execute all aspect reports in parallel, I use fai.fork(). The fork takes the initial universe description, maps it into many targets (via universe_details_mapper), and then reduces them into a single coherent report. This composable structure is both readable and scalable — adding new layers of complexity doesn’t increase coupling.

This multi-step generation strategy beats a one-shot prompt in quality. Prompting the model with focused tasks like “describe the government system” or “describe the role of AI in society” leads to more thoughtful and diverse responses. If you asked for 1000 words directly, you'd likely get vague or generic content.

Maintaining coherence is a challenge, though. Since the reports are generated in parallel, they don’t see each other. In future versions, I may run a post-pass coherence check or give the model the option to refine earlier outputs. But in practice, the shared context from the initial universe seed keeps the tone and world-building surprisingly aligned.

Finally, the full setting is persisted using fai.store(), so it can be reused during story generation, task formulation, and user feedback. This saves compute and gives a stable base for the session.

And yes — in practice, things do fail occasionally. That’s why the retry mechanism via fai.catch() is critical. It ensures that even if one aspect generation fails or times out, we can still move forward with a solid fallback.

## Parallel Execution in Functional Programming

Certain parts of the AI tutor require merging multiple pieces of information at once — like combining the setting and practice rule to generate a task, or fusing the task and response to provide feedback. Doing these steps sequentially adds unnecessary delay, especially when each step involves a network-bound LLM call. That’s where the parallel() operator comes in: it allows multiple AI calls to run concurrently and efficiently, without blocking each other.

Rather than implementing parallel logic inline every time, I built parallel() as a general-purpose operator. Like other parts of this library, it follows a functional pattern: each step is self-contained, stateless, and easy to test in isolation. This reuse saves a lot of boilerplate and helps keep the architecture clean.

Internally, parallel() uses ThreadPoolExecutor, just like fork(). However, here it's used to run a fixed set of targets passed in as a list. Each target is submitted as a job to the thread pool, and once all jobs complete, their results are passed to a reducer function for final assembly. This pattern works especially well for tasks like ai_parallel(), where multiple sources (e.g. profile + task + rule) need to be combined before prompting the LLM again.

```python
def parallel(targets: list[Target], reducer, key: str = None):
    class Parallel(Target):
        def __init__(self):
            super().__init__(key=key)
            self.targets = targets
            self.reducer = reducer

        def __call__(self, *args, **kwargs):
            with ThreadPoolExecutor() as executor:
                results = [executor.submit(target, *args, **kwargs) for target in targets]
                results = [f.result() for f in results]
                results = {target.key: result for target, result in zip(self.targets, results)}

            return self.reducer(**kwargs, **results)

    return Parallel()

def ai_parallel(template, targets: list[Target], llm: str = None, tools: list = None, key: str = None):
    def reducer(**kwargs):
        return infer(template, llm, tools)(**kwargs)
    return parallel(targets, reducer, key)
```

The operator takes three main arguments: targets, a list of callable Target instances; reducer, a function that accepts all the target results as keyword arguments; and an optional key to label the resulting operator. The reducer gives full control over how to combine the results — it might format a prompt, apply postprocessing, or route the result to another tool.

The ai_parallel() function is a lightweight wrapper around parallel(), specialized for AI use. It assumes the reducer’s job is to build a prompt using all the results and then pass that prompt to infer() with optional LLM and tool parameters. This makes it dead simple to build multi-input AI tasks without writing custom glue every time.

```python
def test_parallel():
    def reducer(one, two, three):
        return f"{one}: One | {two}: Two | {three}: Three"

    join_target = parallel(
        targets=[
            dummy(template="One", key="one"),
            dummy(template="Two", key="two"),
            dummy(template="Tree", key="three")
        ],
        reducer=reducer
    )

    result = join_target()
    assert result == "One: One | Two: Two | Tree: Three", \
        f"Expected 'One: One | Two: Two | Tree: Three', got {result}"
```

This operator plays a key role in the overall architecture of the Sci-Fi tutor. It's used for both the task provider and the response examiner — both of which require fusing multiple dynamic values together. Instead of nesting or chaining steps, we run them in parallel and cleanly merge the outcome. That leads to faster runtime, cleaner logic, and easier debugging.

## Story, Rule, Task, and Examiner

The story isn’t just a backdrop — it’s an engine for immersion and structured learning. By splitting the story into paragraphs, I let users consume it in small, digestible pieces. Each paragraph becomes an anchor point for the session, guiding what the user reads, thinks about, and responds to. This gradual exposure helps reinforce vocabulary and comprehension, without overwhelming the user with a wall of text.

To manage the flow, I use a story_end tag that marks when we’ve exhausted the story content. After each paragraph is extracted, the loop checks whether there’s more to read. This design avoids guessing story length or hardcoding limits. When the tag appears, we simply end the loop gracefully.

```python
def story_prompt_template(it: str) -> str:
    return f"""
Create a story for the Sci-Fi universe based on the setting.
The story should be approximately 1000 words long and should be split into paragraphs.
Each paragraph should be approximately 2-3 sentences long.
----------------------------------------------------------------------------
Setting:\n\n{it}"""

story_end = "<story_end>"

def story_paragraph_template(it: str, paragraph_no: int) -> str:
    return f"""
Extract paragraph {paragraph_no} from the story.
If the paragraph does not exist, return an {story_end} tag.
Paragraphs are numbered starting from 0.
----------------------------------------------------------------------------
Story:\n\n{it}"""

def practice_rule_template() -> str:
    return f"""
Based on the profile (use load_profile tool), pick a rule to practice.
The rule should assess the user's mastery of the English language.
It could be a grammar exercise, vocabulary test, or comprehension question.
If the profile is not available, start evaluating user's abilities from scratch.
The rule should be clear and concise, explained briefly.
Do not provide an exercise or question, just the rule itself.
----------------------------------------------------------------------------
<Examples are omitted for brevity>
----------------------------------------------------------------------------
"""

def task_provider_template(sett, rule) -> str:
    return f""""
Create a task based on the setting and the practice rule.
The task should be related to the Sci-Fi universe,
And should require the user to apply the knowledge from the practice rule.
The task should be engaging and encourage the user to think creatively.
The task should be relatively short, so the user can answer in one sentence.
----------------------------------------------------------------------------
Setting:\n\n{sett}\n\n
----------------------------------------------------------------------------
Practice Rule:\n\n{rule}
"""

def response_examiner_template(rule, task, response) -> str:
    return f"""
Examine the user's response to the task.
Provide feedback on the response, including any errors or areas for improvement.
Load and update the user's profile to reflect the user's current abilities.
If the response is correct, provide positive feedback and suggest further practice.
----------------------------------------------------------------------------
Practice Rule:\n\n{rule}\n\n
----------------------------------------------------------------------------
Task:\n\n{task}\n\n
----------------------------------------------------------------------------
User's Response:\n\n{response}
"""
```

The practice rule is a key piece of personalization. It’s not static or pre-written — instead, it’s dynamically generated each time using the user's evolving profile. The rule might suggest practicing tense consistency, avoiding filler words, or using conditional statements — all based on recent performance. This keeps the learning focused and relevant.

Once the rule is in place, the system generates a task that blends the Sci-Fi setting with that specific rule. It might ask the user to describe how a character uses a device, explain a fictional event, or speculate about alien behavior — all while applying the rule in practice. Then, the response is examined by the AI itself, which gives feedback and updates the profile. In this way, the loop becomes self-adjusting.

Both the rule generation and the examiner steps use load_profile and update_profile tools. These allow the system to “remember” what the user is struggling with, and adapt accordingly. The profile lives outside the main loop, so it can persist across sessions if needed.

```python
story = fai.store(
    fai.transform(story_prompt_template, target=setting), filename=".story")
story_paragraph = fai.transform(story_paragraph_template, target=story)

practice_rule = fai.cache(
    fai.infer(practice_rule_template, tools=[load_profile]), key="rule")

task_provider = fai.cache(
    fai.ai_parallel(
        task_provider_template, targets=[setting, practice_rule]), key="task")

response_examiner = fai.ai_parallel(
    response_examiner_template,
    targets=[practice_rule, task_provider],
    tools=[update_profile, load_profile])
```

Behind the scenes, cache() and ai_parallel() play an important role. cache() ensures that once a rule or task is generated, it doesn't need to be re-inferred on the same input. ai_parallel() fuses multiple inputs (like setting + rule, or rule + task + response) into a single reducer call, optimizing both performance and modularity.

Updating the user’s profile after each response is crucial. It’s how the tutor becomes personalized — not by fixed difficulty levels, but by responding to how well the user is currently doing. Over time, the feedback gets more tailored, more challenging, and more useful.

All this makes the experience feel more like a dialogue than a quiz. The system doesn’t test for right answers; it nudges the user forward, in a world that’s interesting enough to keep learning fun, but challenging.

## The AI Tutor: A Sci-Fi Adventure

All components of the AI tutor come together in a simple but immersive command-line loop. I chose the CLI intentionally — not just for ease of development, but because it strips away distractions. There’s no GUI to click through, no visuals to manage. You just enter a terminal and talk to your tutor, like it’s a shipboard AI on some interstellar cruiser.

The loop structure directly reflects the architecture we've built: extract a paragraph → present a rule → generate a task → wait for response → evaluate it. Each pass through the loop advances both the story and the user's learning. It’s narrative-driven, adaptive, and repeatable — a cycle of discovery and practice.

Progress through the story is tracked via paragraph_no, starting at 0. This keeps things deterministic and lets the tutor give the story in small, digestible doses. It also introduces a natural pacing rhythm — each paragraph is a scene, each rule a new challenge. When the AI outputs the special <story_end> tag, the loop gracefully shuts down, signaling the session's narrative conclusion.

```python
if __name__ == '__main__':
    async def run_agent():
        await get_backend().create_session()

        print("Welcome to the Sci-Fi Universe Agent!")
        paragraph_no = 0

        while True:
            paragraph = story_paragraph(paragraph_no=paragraph_no)
            if story_end in paragraph:
                print("The story has ended.")
                break
            print_blue(paragraph)
            print_dash()

            rule = practice_rule()
            print_yellow(rule)
            print_dash()

            task = task_provider()
            print_yellow(task)
            print_dash()

            user_response = input()
            if user_response.lower() in ['exit', 'quit']:
                print("Exiting the agent.")
                break

            feedback = response_examiner(response=user_response)
            print_green(feedback)
            print_dash()

            paragraph_no += 1
            practice_rule.clear()
            task_provider.clear()

    asyncio.run(run_agent())
```

To improve the terminal experience, I added formatting functions like print_blue(), print_yellow(), and print_green() to distinguish between story content, instructions, and feedback. This color-coding helps the user orient themselves in the loop — where they are and what’s expected next.

The interaction flow is simple: the user reads a paragraph aloud, sees a rule they should focus on, receives a task set in the universe, and answers with a single sentence. Then they get instant feedback — not a score, but constructive suggestions based on the rule and their response. It feels conversational, not evaluative.

After each loop, both practice_rule and task_provider are explicitly cleared using .clear(). This is crucial for cache invalidation — otherwise, the same rule and task would repeat. Clearing them ensures the next iteration starts fresh and builds on the updated user profile.

The test suite (test_setting(), test_story(), etc.) is how I verify that each stage of the loop behaves as expected. These aren’t full integration tests, but focused probes that help catch bugs early — like an invalid prompt, a formatting issue, or a misbehaving reducer. Each test runs the AI target asynchronously with mock inputs to validate that the building blocks work in isolation.

```python
def test_setting():
    async_llm_test(setting)

def test_story():
    async_llm_test(story)

def test_story_paragraph():
    async_llm_test(story_paragraph, paragraph_no=0)

def test_practice_rule():
    async_llm_test(practice_rule)

def test_task_provider():
    async_llm_test(task_provider)

def test_response_examiner():
    async_llm_test(response_examiner, response="This is a test response.")
```

The Agent gives the illusion of an intelligent companion — a personality. It doesn’t just deliver content; it reacts, encourages, challenges, and evolves with the user. By weaving story and learning into one flow, it creates a sense of presence that a worksheet never could.

And that is all for this article, folks!

***

I hope you liked the article as much as I did writing the code for it. If you need any help with code or projects or have any questions, feel free to reach out. I would be happy to help. If you need the full version of the code with all the imports, please ping me via DM or in the comments. I also post on X @ https://x.com/gzozulin to update periodically about the project and share new ideas.