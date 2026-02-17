
import torch
import numpy as np
from training_config import glob as g


class iopairs:

    def get_batch(split):
        if split == 'train':
            data = np.memmap('train.bin', dtype = np.uint16, mode = 'r')
        else:
            data = np.memap('validation.bin', dtype = np.uint16, mode = 'r')

        ix = torch.randint(len(data) - g.block_size, (g.batch_size,)) #black_size is the context window
        x = torch.stack([torch.from_numpy((data[i:i+g.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+g.block_size]).astype(np.int64)) for i in ix])

        if g.device_type == 'cuda':
            x, y = x.pin_memory().to(g.device, non_blocking = True), y.pin_memory().to(g.device, non_blocking = True)
        else:
            x, y = x.to(g.device), y.to(g.device)
        return x, y