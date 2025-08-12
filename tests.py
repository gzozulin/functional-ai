from operators.cache import test_cache
from operators.catch import test_catch
from operators.extract import test_extract
from operators.agent import test_ai_agent, test_ai_agent_2
from operators.loop import test_loop
from operators.parallel import test_parallel
from operators.sequential import test_sequential, test_sequential_2
from operators.switch import test_switch
from operators.transform import test_transform

def test_all():
    test_ai_agent()
    test_ai_agent_2()
    test_transform()
    test_cache()
    test_catch()
    test_extract()
    test_loop()
    test_switch()
    test_parallel()
    test_sequential()
    test_sequential_2()

