import torch.nn as nn
from fcsn import FCSN_ENC_SD, FCSN_MID_SD
import torch.optim as optim
from config import factory
from keyFrameSelector import SK
class SD(nn.Module):

    def __init__(self):
        super(SD, self).__init__()

        self.n_feature = 1024

        self.enc = nn.Sequential(
            FCSN_ENC_SD(),
            FCSN_MID_SD(n_class=self.n_feature)
        )
        self.deconv1 = nn.ConvTranspose1d(
            self.n_feature, self.n_feature, 4, padding=1, stride=2, bias=False)
    
        self.deconv2 = nn.ConvTranspose1d(
            self.n_feature, self.n_feature, 16, stride=16, bias=False)
        
        self.avgpool = nn.AvgPool1d(2, stride=2,ceil_mode=True)
        #self.fc = nn.Linear(self.n_feature, 1, bias=False)
        self.fc = nn.Conv1d(self.n_feature, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x

        h = self.enc(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
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
    opt = optim.Adam(net.parameters(), lr=0.0002)
    loss = nn.BCELoss()
    loaders = factory.get_train_loaders()

    netG = SK().to('cuda')
    optG = optim.Adam(netG.parameters(), lr=0.001)

    for i, batch in enumerate(loaders):
        v = batch[0].to('cuda')
        s = batch[1].to('cuda')
        opt.zero_grad()
        
        
        
        s_f, p = netG(v)
        out2 = net(s_f[:,:,p])
        l2 = loss(out2, torch.zeros(1,1).to("cuda")) 
        l2.backward()
        opt.step()

        out = net(s)
        l = loss(out, torch.ones(1, 1).to('cuda')) 
        l.backward()
        opt.step()
        print(l.item(), l2.item(), out, out2)
    

  