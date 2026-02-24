import torch
from .. import expose, run_simple_server


class TestServer(object):
    @expose()
    def get(self, i):
        # return torch.tensor([i]).long()
        # return torch.ones(1000, dtype=torch.long) * i
        return torch.ones(256, 256, 3*2, dtype=torch.long) * i
    
    @expose()
    def num(self):
        return 100


if __name__ == "__main__":
    test_server = TestServer()
    run_simple_server(
        test_server,
        uri_name="test_server",
        daemon_port=24813
    )
