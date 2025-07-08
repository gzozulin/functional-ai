from auxiliary import accepted_keys, safe_lambda
from operators.target import Target

def dummy(template, key=None):
    class Dummy(Target):
        def __init__(self):
            super().__init__(key=key)
            self.call = template
            self.call_keys = accepted_keys(template)

        def __call__(self, *args, **kwargs):
            if callable(self.call):
                return safe_lambda(self.call, self.call_keys, *args, **kwargs)
            return template
    return Dummy()
