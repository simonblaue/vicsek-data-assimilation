from dataclasses import dataclass, field

class Test():

    def __init__(self, config):
        self.config=config
        print(config.h)
        pass



@dataclass
class TestData:

    exec_ref = Test
    h = 'hallo'

c = TestData.exec_ref(TestData)