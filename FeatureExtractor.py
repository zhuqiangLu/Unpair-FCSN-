'''
the original author of this code is https://github.com/pcshih

the origin code: https://github.com/pcshih/pytorch-VSLUD/blob/master/training_set_preparation/FeatureExtractor.py

'''

import torchvision
import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self):

        # supposed input format(N,C,L) C:#features L:#frames
        super(FeatureExtractor, self).__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # torchvision0.3.0
        self.googlenet = torchvision.models.googlenet(pretrained=True)
        # use eval mode to do feature extraction
        self.googlenet.eval()

        # we only want features no grads
        for param in self.googlenet.parameters():
            param.requires_grad = False

        # feature extractor
        self.model = nn.Sequential(*list(self.googlenet.children())[:-2])

        #self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def forward(self, x):
        # put data in to device
        x = x.to(self.device)

        h = self.model(x)

        # the shape of h is (n_frame, 1024, 1, 1)

        h = h.view(h.size()[0], 1024)
        h = h.transpose(1, 0)

        return h


if __name__ == '__main__':
    data = torch.randn((320, 3, 224, 224))
    print(data.shape)
    net = FeatureExtractor()
    print(net(data).shape)
