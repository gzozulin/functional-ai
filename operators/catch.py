import time

from operators.agent import Agent, simple_agent

def catch(agent: Agent, exception: Agent):
    class Catch(Agent):
        def __call__(self, *args, **kwargs):
            try:
                return agent(*args, **kwargs)
            except Exception as e:
                return exception(*args, **kwargs, error=e)

        @property
        def key(self):
            return agent.key

    return Catch()

def retry(agent: Agent, timeout_millis: int = 1000, timeout_mult: int = 1, max_retry: int = 3):
    class Retry(Agent):
        def __init__(self):
            super().__init__()
            self.timeout_millis = timeout_millis
            self.timeout_mult = timeout_mult
            self.max_retry = max_retry
            self.iteration = 0

        def __call__(self, *args, **kwargs):
            while True:
                try:
                    return agent(*args, **kwargs)
                except Exception as e:
                    timeout = self.timeout_millis * self.timeout_mult * self.iteration
                    seconds = timeout / 1000.0
                    time.sleep(seconds)

                    self.iteration += 1
                    if self.iteration > self.max_retry:
                        raise e

        @property
        def key(self):
            return agent.key

    return Retry()

def test_catch():
    def agent_func():
        raise ValueError("An error occurred")

    def handle_func(error):
        return f"Handled error! {error}"

    result = catch(
        agent=simple_agent(call=agent_func),
        exception=simple_agent(call=handle_func))()

    assert "Handled error!" in result, "Catch should handle the error and return the handled value"

