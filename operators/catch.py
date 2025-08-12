from operators.agent import Agent, simple_agent

def catch(agent: Agent, exception: Agent):
    class Catch(Agent):
        def __call__(self, *args, **kwargs):
            try:
                return agent(*args, **kwargs)
            except Exception as e:
                return exception(*args, **kwargs, error=e)

    return Catch()

def test_catch():
    def agent_func():
        raise ValueError("An error occurred")

    def handle_func(error):
        return f"Handled error! {error}"

    result = catch(
        agent=simple_agent(call=agent_func),
        exception=simple_agent(call=handle_func))()

    assert "Handled error!" in result, "Catch should handle the error and return the handled value"

