from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL_GPT_4O = "openai/gpt-4o"
MODEL_GPT_4O_MINI = "openai/gpt-4o-mini"
MODEL_DEEPSEEK_CHAT = "deepseek/deepseek-chat"

APP_NAME = "fun_ai"
USER_ID = "12345"
SESSION_ID = "11223344"

def print_debug(text):
    print(f"\033[93m{text}\033[0m")

class GoogleAdkBackend:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.session = None

    async def create_session(self):
        self.session = await self.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    def create_runner(self, llm: str = None, tools=None, schema=None):
        agent = LlmAgent(
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
            agent=agent,
            app_name=APP_NAME,
            session_service=self.session_service)

        return agent, runner

    @staticmethod
    def print_event(event: Event):
        for part in event.content.parts:
            if part.function_call is not None:
                print_debug(f'Tool call >>> {part.function_call.name}({part.function_call.args})')
            elif part.function_response is not None:
                print_debug(f'Tool response >>> {part.function_response.response["result"][:100]}...')
            elif part.text is not None:
                print_debug(f'LLM text >>> {part.text[:100]}...')

    @staticmethod
    def call_agent(query: str, runner) -> str:
        content = types.Content(role='user', parts=[types.Part(text=query)])
        for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            GoogleAdkBackend.print_event(event)
            if event.is_final_response():
                if event.content and event.content.parts:
                    return event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    raise RuntimeError(f"Agent escalated: {event.error_message or 'No specific message.'}")
                break

        raise RuntimeError("No final response received from the agent.")

backend = GoogleAdkBackend()  # Create a singleton instance of the backend

def get_backend():
    return backend
