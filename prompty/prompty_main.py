import os

import operators as fai
from auxiliary import print_llm_blue, print_success_green
from prompts import PromptBuilder

prompts_dir = 'system-prompts-and-models-of-ai-tools'

def access_prompt(dir: str, file: str) -> str:
    """Access a prompt file from the prompts directory.
    Args:
        dir (str): The directory name within the prompts directory.
        file (str): The filename of the prompt to access.
        Returns: str: The content of the prompt file or an error message if not found.
    """

    path = os.path.join(prompts_dir, dir, file)
    if not os.path.exists(path):
        return "Prompt not found."

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def test_access_prompt():
    print_success_green(access_prompt('Cluely', "Default Prompt.txt"))

def prompts_list() -> str:
    result = "Available example prompts:\n"
    dirs = os.listdir(prompts_dir)
    for dir in dirs:
        dir_full = os.path.join(prompts_dir, dir)
        if '.git' in dir_full:
            continue

        if os.path.isdir(dir_full):
            result += f"{dir}\n"
            files = os.listdir(os.path.join(prompts_dir, dir))
            for file in files:
                result += f"- {file}\n"
    return result

def test_prompts_list():
    print_success_green(prompts_list())

def interviewer_template(user_input: str, chat_history: list[str]) -> str:
    return (PromptBuilder()
     .text("You are Prompty, an agent who helps creating prompts.").dash()
     .text("The User will provide the topic of the prompt and a brief description.")
     .text("Your goal is to help to create the final prompt to the LLM").dash()
     .num("Start by introducing 10-15 instructions that should be covered in the prompt").tab()
     .point("Should be simple numbered points with minimal amount of text")
     .point("Examples: agent role, agent task, style of communication, etc").back()
     .num("The user will pick which instructions are the most relevant and should be covered")
     .num("Start with the first instruction, come up with 5-7 questions to cover")
     .num("The user will pick which answers are the most relevant")
     .num("Based on the topic in User input and selected questions, create a paragraph for this instruction").tab()
     .point("This should be a consize paragraph, each selected answer should be covered in one sentence")
     .point("Use simple and clear language")
     .point("Avoid jargon and technical terms, ambiguity, and unnecessary complexity")
     .point("The paragraph should be self-contained and not refer to other instructions").back()
     .num("For each instruction, come up with an example: <example> ... </example>").tab()
        .point("An example should contain a condition and an expected outcome")
     .num("The user will mention if they want to introduce corrections or move on to the next instruction")
     .num("Once the instruction is covered, move to the next instruction")
     .num("When all instructions are covered, print a single stop word: !done").dash()
     .text("The topic and the brief description from the User:").text(user_input).dash()
     .chat(chat_history)
     .prompt)

def test_interviewer_template():
    print_success_green(interviewer_template(
        user_input="Create a prompt for an AI assistant that helps users with their daily tasks.",
        chat_history=["What is the main role of the AI assistant?"]))

def prompt_template(interview: str) -> str:
    return (PromptBuilder()
    .text("You are Prompty, an agent who helps creating prompts.")
    .text("Based on the Interview with the User, create a final prompt to the LLM").dash()
    .num("With the tool at your disposal, select the most relevant example prompt")
    .tab().point("Choose wisely: do not call more than 3 example prompts to avoid overwhelming the context").back()
    .num("Mimic the style and structure of the example prompt")
    .num("The prompt format should be Markdown for better readability")
    .tab().point("Use headings, bullet points, and other Markdown elements where appropriate").back()
    .num("Use second-person imperative (or direct address) for the prompt: 'You are...', 'Your task is...'")
    .num("Include examples for each instruction in the prompt")
    .num("Print only the final prompt, nothing else").dash()
    .text(prompts_list()).dash()
    .text("Example Final Prompt:").file("static/example").dash()
    .text("Interview with the User:").nl()
    .tag_open("interview").text(interview).tag_close()
    .prompt)

def test_prompt_template():
    print_success_green(prompt_template(interview="This is a sample interview text."))

interview_agent = fai.transform(
    agent=fai.ai_chat(
        agent=fai.retry(
            agent=fai.ai_agent(template=interviewer_template)),
        output_llm=print_llm_blue, input_user=input),
    transformer=lambda it: PromptBuilder().chat(it).prompt,
    key="interview")

prompty_agent = fai.retry(fai.ai_transform(
    template=prompt_template,
    agent=interview_agent,
    tools=[access_prompt]))

if __name__ == "__main__":
    filename = input("Please enter the filename: ")
    prompty_result = prompty_agent(
        user_input=PromptBuilder().file(filename).prompt)
    print_success_green(prompty_result)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompty_result)
