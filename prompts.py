import textwrap

abc = "abcdefghijklmnopqrstvuwxyz"

class PromptBuilder:
    def __init__(self):
        self.prompt = ""
        self.tabs = 0
        self.number = 0
        self.letter = 0
        self.tag_stack = []

    @property
    def __indent(self):
        return '\t' * self.tabs

    def nl(self):
        self.number = 0
        self.letter = 0
        self.prompt += "\n"
        return self

    def tab(self):
        self.tabs += 1
        return self

    def back(self):
        self.tabs = max(0, self.tabs - 1)
        return self

    def text(self, text: str):
        self.prompt += textwrap.indent(textwrap.dedent(text), self.__indent)
        self.nl()
        return self

    def dash(self, symb='-', cnt=50):
        self.prompt += textwrap.indent(f"{cnt * symb}", self.__indent)
        self.nl()
        return self

    def point(self, text: str, symb='*'):
        self.prompt += textwrap.indent(f'{symb} {text}', self.__indent)
        self.prompt += "\n"
        return self

    def num(self, text: str):
        self.number += 1
        self.prompt += textwrap.indent(f'{str(self.number)}. {text}', self.__indent)
        self.prompt += "\n"
        return self

    def let(self, text: str):
        ltr = abc[self.letter]
        self.letter += 1
        self.letter = min(self.letter, len(abc))
        self.prompt += textwrap.indent(f'{ltr}. {text}', self.__indent)
        self.prompt += "\n"
        return self

    def file(self, filename: str):
        with open(filename, 'r') as f:
            content = f.read()
        self.prompt += textwrap.indent(content, self.__indent)
        self.nl()
        return self

    def substitute(self, old: str, new: str):
        self.prompt = self.prompt.replace(old, new)
        return self

    def chat(self, messages: list[str], first: str = 'LLM:', second: str = 'User:'):
        is_first = True
        for message in messages:
            if is_first:
                self.text(first)
            else:
                self.text(second)
            self.text(message)
            is_first = not is_first
        return self

    def list(self, items: list[str], symb='*'):
        for item in items:
            self.point(item, symb)
        return self

    def tag_open(self, tag: str):
        self.tag_stack.append(tag)
        self.prompt += textwrap.indent(f"<{tag}>", self.__indent)
        self.nl()
        return self

    def tag_close(self):
        if self.tag_stack:
            tag = self.tag_stack.pop()
            self.prompt += textwrap.indent(f"</{tag}>", self.__indent)
            self.nl()
        return self

def test_prompt_builder():
    print(PromptBuilder().nl()
        .dash().text("Hello, this is a professional prompt builder!").nl()
        .dash().text("We will start by adding a few points:").tab()
        .point("This is the first point!")
        .point("You guessed it: this is the second point!")
        .point("And this is the third point!").back()
        .dash().text("Now lets jump into the numbers:").tab()
        .num("This is the first number!").tab()
        .let("We can have a few letters as well")
        .let("As many as ABC has!").back()
        .num("You guessed it: this is the second number!")
        .num("And this is the third number!").back()
        .dash().file('prompts.py')
        .dash().chat(["Hello, how are you?", "I'm fine, thank you! How about you?"])
        .dash().list(["First item", "Second item", "Third item"])
        .tag_open("div")
        .text("This is inside a div tag")
        .tag_close()
        .prompt)
