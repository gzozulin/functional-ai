from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL_GPT_4O = "openai/gpt-4o"
MODEL_GPT_4O_MINI = "openai/gpt-4o-mini"

APP_NAME = "fun_ai"
USER_ID = "12345"
SESSION_ID = "11223344"

class GoogleAdkBackend:
    def __init__(self):
        self.session_service = InMemorySessionService()

    async def create_session(self):
        await self.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    def create_runner(self, llm: str = None, tools=None):
        target_agent = LlmAgent(
            model=LiteLlm(model=llm),
            name="functional_ai_agent",
            instruction="You are a helpful assistant",
            description="An agent that performs tasks based on instructions",
            tools=tools)

        runner = Runner(
            agent=target_agent,
            app_name=APP_NAME,
            session_service=self.session_service)

        return runner

    @staticmethod
    def call_agent(query: str, runner) -> str:
        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."  # Default
        for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break

        return final_response_text

backend = GoogleAdkBackend()  # Create a singleton instance of the backend

def get_backend():
    return backend
