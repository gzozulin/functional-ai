from operators import dummy
from operators.target import Target

def catch(target: Target, exception: Target):
    class Catch(Target):
        def __call__(self, *args, **kwargs):
            try:
                return target(*args, **kwargs)
            except Exception as e:
                return exception(*args, **kwargs, error=e)

    return Catch()

def test_catch():
    def target_func():
        raise ValueError("An error occurred")

    def handle_func(error):
        return f"Handled error! {error}"

    result = catch(
        target=dummy(template=target_func),
        exception=dummy(template=handle_func))()

    assert "Handled error!" in result, "Catch should handle the error and return the handled value"

