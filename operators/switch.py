from operators.dummy import dummy
from operators.target import Target

def switch(ifbranch: Target, elsebranch: Target, condition, key: str = None):
    class Switch(Target):
        def __init__(self):
            super().__init__(key=key)
            self._ifbranch = ifbranch
            self._elsebranch = elsebranch

        def __call__(self, *args, **kwargs):
            if condition(**kwargs):
                return self._ifbranch(*args, **kwargs)
            else:
                return self._elsebranch(*args, **kwargs)

    return Switch()

def test_switch():
    if_branch = dummy(template="This is the if branch")
    else_branch = dummy(template="This is the else branch")

    condition_true = lambda **kwargs: True
    condition_false = lambda **kwargs: False

    switch_true = switch(ifbranch=if_branch, elsebranch=else_branch, condition=condition_true)
    result_true = switch_true()
    assert result_true == "This is the if branch", f"Expected 'This is the if branch', got {result_true}"

    switch_false = switch(ifbranch=if_branch, elsebranch=else_branch, condition=condition_false)
    result_false = switch_false()
    assert result_false == "This is the else branch", f"Expected 'This is the else branch', got {result_false}"
