import torch
from torch import nn

from bgpsyche.stage3_rank.make_dataset import make_as_level_dataset

input_size = 28

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()



def _test():
    dataset = make_as_level_dataset()


if __name__ == '__main__': _test()