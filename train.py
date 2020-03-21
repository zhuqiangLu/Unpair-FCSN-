from keyFrameSelector import SK
from summaryDiscriminator import SD
from dataloader import get_dataloader
import torch.optim as optim
import torch.nn as nn
import config
import torch


class Trainer(object):

    def __init__(self, data_path, alpha, beta, n_class=2):
        self.dataloader = get_dataloader()

        self.SD = SD()
        self.opt_SD = optim.SGD(self.SD.parameters(), lr=config.SD_lr)
        self.SD_losses = list()

        self.SK = SK()
        self.opt_SK = optim.Adam(self.SD.parameters(), lr=config.SK_lr)
        self.SK_losses = list()

        self.device = torch.device("cuda:0" if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.crit_adv = nn.BCELoss()

        self.epoch = config.epoch
        self.alpha = alpha
        self.beta = beta

    def crit_reconst(self, pred_sum, gt_sum):
        '''the expected dim of pred is [1, c, t]'''
        k = pred.shape(2)
        return torch.norm(pred_sum-gt_sum, dim=2).sum()/k

    def crit_div(self, pred, gt):
        '''the expected dim of pred is [1, c, t]'''
        k = pred.shape(2)
        cos = nn.CosineSimilarity(dim=0)
        loss = torch.zeros([0])
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                else:
                    loss += cos(pred[0, :, i], gt[0, :, j])
        return loss/(k*(k+1))

    def train(self):
