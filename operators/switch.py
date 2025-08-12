from auxiliary import accepted_keys, safe_lambda
from operators.agent import Agent, simple_agent

def switch(ifbranch: Agent, elsebranch: Agent, condition, key: str = None):
    class Switch(Agent):
        def __init__(self):
            super().__init__(key=key)
            self._ifbranch = ifbranch
            self._elsebranch = elsebranch
            self._condition = condition
            self._condition_keys = accepted_keys(condition)

        def __call__(self, *args, **kwargs):
            if safe_lambda(self._condition, self._condition_keys, **kwargs):
                return self._ifbranch(*args, **kwargs)
            else:
                return self._elsebranch(*args, **kwargs)

    return Switch()

def test_switch():
    if_branch = simple_agent(call="This is the if branch")
    else_branch = simple_agent(call="This is the else branch")

    condition_true = lambda **kwargs: True
    condition_false = lambda **kwargs: False

    switch_true = switch(ifbranch=if_branch, elsebranch=else_branch, condition=condition_true)
    result_true = switch_true()
    assert result_true == "This is the if branch", f"Expected 'This is the if branch', got {result_true}"

    switch_false = switch(ifbranch=if_branch, elsebranch=else_branch, condition=condition_false)
    result_false = switch_false()
    assert result_false == "This is the else branch", f"Expected 'This is the else branch', got {result_false}"
