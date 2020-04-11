import torch.nn as nn
from fcsn import FCSN
import torch


class SK(nn.Module):
    def __init__(self, ratio=0.15, n_class=1024, T=320):
        super(SK, self).__init__()

        self.ratio = ratio
        self._SK = FCSN(n_class=n_class)  # backbone fcsn
        self.score_layer = nn.Sequential(  # to score the feature based on the decoded features
            nn.Conv1d(n_class, 1, 1),
            nn.Sigmoid(),
        )

        self.conv1 = nn.Conv1d(n_class, n_class, 3, padding=1)

    def forward(self, x):
        video_features = x

        k = int(x.shape[2] * self.ratio)

        h = self._SK(x)

        sk_out = h  # sk_out is the decoded features

        scores = self.score_layer(h)  # shape (1,1,T)


        #this part not in the gradient graph
        '''
            start
        ''' 
        top_scores = torch.topk(scores.detach(), k, dim=-1)
        high = top_scores.values[0, 0, 0].data.cpu().detach()
        low = top_scores.values[0, 0, -1].data.cpu().detach()
        '''
            end
        '''


        scores = torch.where(scores >= low, torch.ones(
            scores.shape), torch.zeros(scores.shape))  # the 1/0 vector

        h = (sk_out * scores)

        '''
            gether non-zeros vectors to form s
        '''
        topk = torch.sum(h, dim=1)  # to remove the zeros rows
        picks = topk.nonzero(as_tuple=True)[1]  # all T that selected as S
        s = h[:, :, picks]  # the selected decoded features

        s = self.conv1(s)  # reconstruct

        s = s + video_features[:, :, picks]
        # print(h[h.nonzero(as_tuple=True)].shape)
        return s, picks


if __name__ == '__main__':
    import torch
    net = SK()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    #print(net.conv1.weight)
    data = torch.randn((1, 1024, 128))
    optimizer.zero_grad()
    h, picks = net(data)
    
    print(picks)
    print(h.shape)

    y = data[0, :, picks].mean(dim=1)
    x = (h[0].mean(dim=1))
    print(x.shape, y.shape)
    loss = torch.mean((y - x)**2) * 1000000
    print(loss.shape)
    print(loss)
    loss.backward()
    optimizer.step()
    # print(net)
    #print(net.conv1.weight)
    # import numpy as np
    # print(np.array(out.cpu().data[0]))
