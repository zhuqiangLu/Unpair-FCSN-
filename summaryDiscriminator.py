import torch.nn as nn
from fcsn import FCSN_ENC, FCSN_MID


class SD(nn.Module):

    def __init__(self):
        super(SD, self).__init__()

        self.n_feature = 2

        self.enc = nn.Sequential(
            FCSN_ENC(),
            FCSN_MID(n_class=self.n_feature)
        )

        self.avgpool = nn.AvgPool1d(2, stride=2)
        self.fc = nn.Linear(self.n_feature, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x

        h = self.enc(h)
        h = self.avgpool(h)

        # shape becomes [n, T, C], becauses of fc
        h = h.view(h.shape[0], -1, self.n_feature)
        h = self.fc(h)
        # sigmoid scores feach frame of the summary
        h = self.sigmoid(h)

        return h


if __name__ == '__main__':
    import torch
    net = SD()
    data = torch.randn((1, 1024, 320))
    out = net(data)
    print(out.shape)
    print(out.mean(dim=1).shape)
    print(torch.zeros(1, 1).shape)
