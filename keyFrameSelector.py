import torch.nn as nn

from fcsn import FCSN


class SK(nn.Module):
    def __init__(self, n_class=2):
        super(SK, self).__init__()
        self._SK = nn.Sequential(FCSN(n_class=n_class), nn.Softmax(dim=1))

    def forward(self, x):
        return self._SK(x)


if __name__ == '__main__':
    import torch
    net = SK(n_class=4)
    data = torch.randn((1, 1024, 320))
    out = net(data)
    print(out.shape)
    print(out[:, :, 1:10])
