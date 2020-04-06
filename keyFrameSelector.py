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

    def forward(self, x):
        video_features = x

        k = int(x.shape[2] * self.ratio)

        h = self._SK(x)

        sk_out = h

        scores = self.score_layer(h)  # shape (1,1,T)

        top_scores = torch.topk(scores, k, dim=-1)
        high = top_scores.values[0, 0, 0].data.cpu()
        low = top_scores.values[0, 0, -1].data.cpu()
        scores = torch.where(scores >= low, torch.ones(
            scores.shape), torch.zeros(scores.shape))
        h = sk_out * scores

        # to remove the zeros rows
        topk = torch.sum(h, dim=1)
        picks = topk.nonzero(as_tuple=True)[1]  # all T that selected as S

        s = h[:, :, picks]

        # print(h[h.nonzero(as_tuple=True)].shape)
        return s


if __name__ == '__main__':
    import torch
    net = SK()
    data = torch.randn((1, 1024, 128))
    h = net(data)
    print(h.shape)
    # import numpy as np
    # print(np.array(out.cpu().data[0]))
