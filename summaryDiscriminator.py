import torch.nn as nn
from fcsn import FCSN_ENC_SD, FCSN_MID_SD
import torch.optim as optim

class SD(nn.Module):

    def __init__(self):
        super(SD, self).__init__()

        self.n_feature = 1024

        self.enc = nn.Sequential(
            FCSN_ENC_SD(),
            FCSN_MID_SD(n_class=self.n_feature)
        )
        
        self.avgpool = nn.AvgPool1d(2, stride=2,ceil_mode=True)
        #self.fc = nn.Linear(self.n_feature, 1, bias=False)
        self.fc = nn.Conv1d(self.n_feature, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x

        h = self.enc(h)
        h = self.avgpool(h)
        
        # shape becomes [n, T, C], becauses of fc
        #h = h.view(h.shape[0], -1, self.n_feature)
        
        h = self.fc(h)
        # sigmoid scores feach frame of the summary
        h = self.sigmoid(h)
        
        return h.mean(dim = 2)

if __name__ == '__main__':
    import torch
    net = SD().to('cuda')
    opt = optim.SGD(net.parameters(), lr=0.002)
    loss = nn.BCELoss()

    for i in range(1):
        # opt.zero_grad()
        data = torch.randn((1, 1024, 19)).to('cuda')
        out = net(data)
        l = loss(out, torch.ones(1, 1).to('cuda'))
        l.backward()
        opt.step()
        print(out.shape)
    

   
    
    
