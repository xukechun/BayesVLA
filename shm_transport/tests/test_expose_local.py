import numpy as np
from .. import expose


class TestServer(object):
    def __init__(self):
        self.data = np.ones(3)
    
    @expose()
    def add_one(self, x):
        return x + 1
    
    @staticmethod
    @expose()
    def add_one_static(x):
        return x + 1
    

if __name__ == "__main__":
    server = TestServer()

    x = np.ones(3)
    y = server.add_one(x)
    # y = server.add_one_static(x)

    print(y)

