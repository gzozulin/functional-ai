from operators.target import Target

def catch(target: Target, handle: Target):
    class Catch(Target):
        def __call__(self, *args, **kwargs):
            try:
                return target(*args, **kwargs)
            except Exception as e:
                return handle(*args, **kwargs, error=e)

    return Catch()
