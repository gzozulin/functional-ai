import asyncio
import os

import operators as fai
from auxiliary import print_yellow, print_blue, print_green, async_llm_test, print_dash
from backends.google_adk import get_backend, MODEL_GPT_4O_MINI
from operators import Target

# --------------- Tools ---------------

def load_profile() -> str:
    """Load the user's profile from the storage."""
    print("Tool: Loading user's profile...")

    if not os.path.exists("user_profile.txt"):
        return "User's profile is not available yet."

    with open("user_profile.txt", "r") as file:
        profile = file.read()
    return profile

def update_profile(profile: str):
    """Update the profile with new information.
    *:param profile: The updated profile content."""
    print("Tool: Updating user's profile...")

    with open("user_profile.txt", "w") as file:
        file.write(profile)

# --------------- Templates ---------------

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
Example of a practice rule:
In English, questions generally follow the structure:
Question word (or auxiliary verb) + subject + verb (+ object/complement).

For example, "Where do you live?" 
Yes/no questions, which don't have a question word, 
Often start with an auxiliary verb: "Do you like pizza?".

Key components of question structure:
Question words (or auxiliary verbs): These are words like "what," "where," 
"when," "who," "why," "how," "do," "is," "are," "can," etc. 
Subject: The person or thing performing the action. 
Verb: The action word. 
Object/Complement: The thing or person affected by the verb,
Or additional information about the subject.

Examples:
"What are you doing?": (Question word: what, Subject: you, Verb: are doing) 
"Is she coming to the party?": (Auxiliary verb: is, Subject: she, Verb: coming) 
"Where does he work?": (Question word: where, Subject: he, Verb: work) 
"How long have you been studying English?": 
    (Question word: how long, Subject: you, Verb: have been studying)
----------------------------------------------------------------------------
Another example of a practice rule:
In English, the past simple tense is used to describe actions that were completed in the past.
The structure is: Subject + past tense verb (+ object/complement).

The past tense verb is often formed by adding -ed to the base form of regular verbs, 
but many common verbs are irregular and have unique past tense forms.

Examples:
"I walked to the store." (regular verb: walk -> walked)
"He went to the store." (irregular verb: go -> went)
The past simple tense is often used with time expressions like:
 "yesterday," "last week," "two days ago," etc.
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

# --------------- Universe ---------------

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

# --------------- Story ---------------

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

# --------------- Tests ---------------

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

# --------------- Main ---------------

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
