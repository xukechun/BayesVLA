import os
import torch
import multiprocessing as mp
from torch.utils.data import Dataset, IterableDataset, DataLoader
from .. import get_shm_proxy
# from .. import safe_exit
# from .. import shm_mng


# mp.Manager().dict()


class IndexDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.client = None
    
    def __len__(self):
        return 100
    
    def __getitem__(self, i):
        if self.client is None:
            self.client = get_shm_proxy(
                uri_name="test_server",
                copy_ret_shm=True
            )
            # print(shm_mng._pid_to_client_shm_managers)
            # print("pid = {}".format(os.getpid()))
            # print(shm_mng.get_client_shm_managers())
            # print(shm_mng._pid_to_client_shm_managers)
            
        # print(shm_mng._pid_to_client_shm_managers)
        # print(mp.current_process().name)
        # print(mp.get_context())
        ret = self.client.get(i)
        return ret


def get_indexed_dataloader(batch_size, num_workers):
    dset = IndexDataset()
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )
    return loader


def test_indexed_dataloader():
    torch.multiprocessing.set_start_method("spawn")

    dataloader = get_indexed_dataloader(20, 4)
    all_data = []
    for data in dataloader:
        all_data.append(data)
    
    all_data = torch.cat(all_data, dim=0)
    print(all_data)
    assert (all_data[1:] >= all_data[:-1]).all()



class IterDataset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        self.proxy = None
    
    def __iter__(self):
        if self.proxy is None:
            self.proxy = get_shm_proxy(
                uri_name="test_server",
                copy_ret_shm=True
            )
        
        num = self.proxy.num()
        for i in range(num):
            data = self.proxy.get(i)
            yield data


def get_iter_dataloader(batch_size, num_workers):
    dset = IterDataset()
    dataloader = DataLoader(
        dataset=dset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True
    )
    return dataloader


def test_iter_dataloader():
    torch.multiprocessing.set_start_method("spawn")

    dataloader = get_iter_dataloader(20, 4)
    for data in dataloader:
        # all_data.append(data)
        assert torch.all(data == data[:, 0:1]), data
    
    # all_data = torch.cat(all_data, dim=0)
    # print(all_data)
    # assert (all_data[1:] >= all_data[:-1]).all()
    # assert torch.all(all_data == all_data[:, 0:1])




if __name__ == "__main__":
    # test_indexed_dataloader()
    test_iter_dataloader()



