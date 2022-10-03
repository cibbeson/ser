from ser.transforms import flip
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def test_flip():
    small_array = np.arange(9).reshape((3,3))
    small_array = torch.tensor(small_array)

    flipped_array = torch.tensor(np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]]))
    test_output = flip()(small_array)
    
    #assert test_output == flipped_array
    assert torch.all(test_output.eq( flipped_array))