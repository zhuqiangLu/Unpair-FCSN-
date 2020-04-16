import torch.nn as nn
from fcsn import FCSN
import torch


class SK(nn.Module):
    def __init__(self, ratio=0.15, n_class=1):
        super(SK, self).__init__()

        self.ratio = ratio
        self._SK = FCSN(n_class=n_class)  # backbone fcsn

        #self.score_layer = nn.Sigmoid()  # to score the feature based on the decoded features
            

        self.conv1 = nn.Conv1d(1, n_class, 3, padding=1)

    def forward(self, x):
        video_features = x

        k = int(x.shape[2] * self.ratio)

        sk_out = self._SK(x)

        scores=sk_out  # sk_out is the decoded features (1, 1, T)

        # scores = self.score_layer(h)  # shape (1,1,T)
        


        #this part not in the gradient graph
        '''
            start
        ''' 
        
        top_scores = torch.topk(scores, k, dim=-1)
        
        high = top_scores.values[0, 0, 0]
        low = top_scores.values[0, 0, -1]
        '''
            end
        '''


        scores = torch.where(scores >= low, torch.ones(
            scores.shape).to('cuda') , torch.zeros(scores.shape).to('cuda'))   # the 1/0 vector
        h = (sk_out * scores)


        '''
            gether non-zeros vectors to form s
        '''
        topk = torch.sum(h, dim=1)  # to remove the zeros rows
        
        picks = topk.nonzero(as_tuple=True)[1]  # all T that selected as S

        h = self.conv1(h)
        s = h + video_features
        
        return s, picks


if __name__ == '__main__':
    import torch
    net = SK().to('cuda')
    data = torch.randn((1, 1024, 64)).to('cuda')
    s, picks = net(data)

    print(s[:,:,picks].shape)
    # net.train()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    # #print(net.conv1.weight)
    # data = torch.randn((1, 1024, 64)).to('cuda')
    # optimizer.zero_grad()
    # h, picks = net(data)
    # picks = picks.cpu().data.numpy()
    # print(picks.shape)
    # print(h.shape)

    # y = data[0, :, picks].mean(dim=1)
    # x = (h[0].mean(dim=1))
    # print(x.shape, y.shape)
    # loss = torch.mean((y - x)**2) * 1000000
    # print(loss.shape)
    # print(loss)
    # loss.backward()
    # optimizer.step()
    # # print(net)
    # #print(net.conv1.weight)
    # # import numpy as np
    # # print(np.array(out.cpu().data[0]))
