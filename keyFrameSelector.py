import torch.nn as nn

from fcsn import FCSN


class SK(nn.Module):
    def __init__(self, n_class=1):
        super(SK, self).__init__()
        self._SK = nn.Sequential(FCSN(n_class=n_class), nn.Sigmoid())

    def forward(self, x):
        return self._SK(x)


if __name__ == '__main__':
    import torch
    net = SK(n_class=1)
    data = torch.randn((1, 1024, 161))
    out = net(data)
    # import numpy as np
    # print(np.array(out.cpu().data[0]))
    print(out.shape)
