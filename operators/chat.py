import operators as fai
from prompts import PromptBuilder

def ai_chat(agent: fai.Agent,
            output_llm: callable = print,
            input_user: callable = input,
            key: str = None,
            stop_word: str = '!done',
            max_iter: int = 100) -> fai.Agent:

    class Chat(fai.Agent):
        def __init__(self):
            super().__init__(key=key)
            self.agent = agent
            self.stop_word = stop_word
            self.max_iter = max_iter

        def __call__(self, *args, **kwargs) -> list[str]:
            chat_history = []
            for _ in range(self.max_iter):
                question = self.agent(chat_history=chat_history, **kwargs)
                chat_history.append(question)
                output_llm(question)

                if self.stop_word in question:
                    break

                reply = input_user()
                chat_history.append(reply)

                if self.stop_word in reply:
                    break

            return chat_history

    return Chat()

def test_ai_chat():
    chat_agent = ai_chat(
        agent=fai.ai_agent(
            template=lambda chat_history:
            PromptBuilder()
            .text("Ask the user about the weather").dash()
            .chat(chat_history).prompt),
        output_llm=print,
        input_user=lambda: "Great, thank you!",
        stop_word='done',
        key="chat_example",
        max_iter=1
    )
    chat = chat_agent()
    assert len(chat) == 2
