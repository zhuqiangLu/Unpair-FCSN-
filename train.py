from keyFrameSelector import SK
from summaryDiscriminator import SD
from utils import score_shot, knapsack, f_score
import torch.optim as optim
import torch.nn as nn
import config
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.variable import Variable

class Trainer(object):

    def __init__(self, beta=1):
        self.factory = config.factory
        self.device = torch.device("cuda:0" if
            torch.cuda.is_available() else "cpu")
        


        self.SD = SD().to('cuda')
        self.opt_SD = optim.Adam(self.SD.parameters(), lr=config.SD_lr)
        self.SD_scheduler = optim.lr_scheduler.StepLR(self.opt_SD, step_size=20, gamma=0.8)



        self.SK = SK().to('cuda')
        self.opt_SK = optim.Adam(self.SK.parameters(), lr=config.SK_lr)
        self.SK_scheduler = optim.lr_scheduler.StepLR(self.opt_SK, step_size=20, gamma=0.8)

        self.crit_adv = nn.BCELoss()

        self.epoch = config.epoch
        self.beta = beta

        self.pred_fake = list()
        self.pred_real = list()
        self.reconst_loss = list()
        self.div_loss = list()
        self.adv_sk = list()
        self.adv_sd_real = list()
        self.adv_sd_fake = list()
        
        
        self.f = dict()

    def crit_reconst(self, pred_sum, gt_sum):
        '''the expected dim of pred is [1, c, t]'''

        k = pred_sum.shape[2]
        return torch.norm(pred_sum-gt_sum, dim=1).sum()/k

    def crit_div(self, pred):
        '''the expected dim of pred is [1, c, t]'''
        k = pred.shape[2]
        cos = nn.CosineSimilarity(dim=0)
        loss = torch.zeros([1]).to("cuda")
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                else:
                    loss += cos(pred[0, :, i], pred[0, :, j])
        return loss/(k*(k+1))


    def real_label(self, N):
        label = Variable(torch.ones(N, 1)).to("cuda")
        return label
    
    def fake_label(self, N):
        label = Variable(torch.zeros(N, 1)).to("cuda")
        return label

    

    def _train(self, v,s):
        '''
            note that this method takes one sample at a time

            make sure the fake_sum are DETACHED from the generator net
        '''

        self.opt_SD.zero_grad()
        self.SD.train()
        for p in self.SD.parameters():
            p.requires_grad = True
        
        '''
            train sd
        '''
    
        
        # train on real data
    
        pred_real = self.SD(s)
        # as all frames belongs to the summary

        sd_loss_real = self.crit_adv(pred_real, self.real_label(s.shape[0]))
        sd_loss_real.backward()

        adv_sd_real_loss = sd_loss_real.item()
        self.opt_SD.step()

        pred_sum, picks = self.SK(v)
        pred_fake= self.SD(pred_sum[:,:,picks].detach())

        sd_loss_fake = self.crit_adv(pred_fake, self.fake_label(s.shape[0])) 
        sd_loss_fake.backward()

        adv_sd_fake_loss = sd_loss_fake.item()
        self.opt_SD.step()

        sd_loss = sd_loss_real + sd_loss_fake
        

        
        
        ''' 
            train sk
        '''

        #pred_sum, picks = self.SK(v)
        self.opt_SK.zero_grad()

        for p in self.SD.parameters():
            p.requires_grad = False
        

        adv_sk_loss = self.crit_adv(self.SD(pred_sum[:,:,picks]), self.real_label(s.shape[0]))
        reconst = self.crit_reconst(pred_sum[:,:,picks], v[:, :, picks],)
        div = (self.beta*self.crit_div(pred_sum[:,:,picks]))
        #print(adv_sk_loss, reconst, div)
        sk_loss = adv_sk_loss + reconst  + div
        sk_loss.backward()
        self.opt_SK.step()

        recosnt_loss = reconst.item()
        adv_sk_loss = adv_sk_loss.item()
        div_loss = div.item()

        sd_pred_fake = pred_fake.cpu().detach().numpy()[0,0]
        sd_pred_real = pred_real.cpu().detach().numpy()[0,0]

        return adv_sd_real_loss,adv_sd_fake_loss,recosnt_loss,adv_sk_loss,div_loss,sd_pred_fake,sd_pred_real
    

        


    def eval(self, features, gt_scores, cps, gt_picks, n_frame_per_seg, n_frame):
        self.SK.eval()

        cps = cps.cpu().data.numpy()[0]
        gt_picks = gt_picks.cpu().data.numpy()[0]
        gt_scores = gt_scores.cpu().data.numpu()[0]
        n_frame_per_seg = n_frame_per_seg.cpu().data.numpy()[0]
        
        _, picks = self.SK(features.to("cuda"))
        picks = picks.cpu().data.numpy()

        pred_scores = np.zeros((gt_picks.shape[0], 1))  # (n, 1)
        pred_scores[picks, :] = 1
        
        pred_seg_scores = score_shot(
            cps, pred_scores, gt_picks, n_frame_per_seg)  # (n_cp, )
        gt_seg_scores = score_shot(
            cps, gt_scores, gt_picks, n_frame_per_seg)  # (n_cp, )

        # the length of the summary
        length = int(n_frame.cpu().data.numpy()[0] * 0.15)

        gt_seg_idx = knapsack(gt_seg_scores, n_frame_per_seg, length)
        pred_seg_idx = knapsack(pred_seg_scores, n_frame_per_seg, length)

        F = f_score(gt_seg_idx, pred_seg_idx, n_frame_per_seg,)
        return F

    def run(self):
    
        
        with trange(self.epoch, position=0) as epoches:

            for epoch in epoches:


                train_loader = self.factory.get_train_loaders()
                dataset_pool = dict()
                avail_data = len(train_loader)
                
                '''
                    train
                '''
                
                for i, batch in enumerate(tqdm(train_loader, position = 1)):
                    v = batch[0].to(self.device)
                    s = batch[1].to(self.device)

                    adv_sd_real_loss,adv_sd_fake_loss,recosnt_loss,adv_sk_loss,div_loss,sd_pred_fake,sd_pred_real = self._train(v, s)

                    self.SD_scheduler.step()
                    self.SK_scheduler.step()

                    self.adv_sk.append(adv_sk_loss )
                    self.reconst_loss.append( recosnt_loss)
                    self.div_loss.append(div_loss)
                    self.adv_sd_real.append(adv_sd_real_loss)
                    self.adv_sd_fake.append(adv_sd_fake_loss)
                    self.pred_fake.append( sd_pred_fake)
                    self.pred_real.append( sd_pred_real)

                    sd_loss = adv_sd_real_loss+adv_sd_fake_loss
                    sk_loss = recosnt_loss+adv_sk_loss+div_loss
                    
                    
                    epoches.set_description('epoch {}, sd_loss {}, sk_loss {}, pred_fake {}, pred_real {}'.format(epoch, sd_loss, sk_loss, sd_pred_fake, sd_pred_real))
                    # train sd with read sum
                
                '''
                    test
                '''

                


                loaders = self.factory.get_test_loaders()
                # keys = list(loaders.keys())
                # for key in keys:
                #     if key not in self.f:
                #         self.f[key] = list()
                f_score = 0
                for i, batch in enumerate(tqdm(loaders)):
                    v = batch[0]
                    s = batch[1]
                    cps = batch[2]
                    gtscore = batch[3]
                    picks = batch[4]
                    n_frame_per_seg = batch[5]
                    n_frames = batch[6]
                    f_score += self.eval(features, gt_seg_scores, cps, picks, n_frame_per_seg, n_frame)
                    counter += 1
                    
                self.f[key].append(f_score/counter)
                    
                


                # with trange(len(keys), position= 2) as idx:
            
                    
                #     for i in idx:
                #         key = keys[i]
                #         loader = loaders[key]
                #         f_score = 0
                #         counter = 0
                #         for j, video_info in enumerate(tqdm(loader)):
                #             features = video_info[0]
                #             gt_seg_scores = video_info[1]
                #             cps = video_info[2]
                #             picks = video_info[3]
                #             n_frame_per_seg = video_info[4]
                #             n_frame = video_info[5]
                #             f_score += self.eval(features, gt_seg_scores, cps, picks, n_frame_per_seg, n_frame)
                #             counter += 1
                            
                #         self.f[key].append(f_score/counter)
                        

        
        self.save_loss_plot()
        self.save_f_plot()
        self.save_pred_plot()


    def save_pred_plot(self):
        plt.clf()

        plt.plot(self.pred_fake, label = 'pred_fake')

        plt.plot(self.pred_real, label = 'pred_real')

        plt.legend()

        plt.savefig('pred.png')

    
    def save_loss_plot(self):
        plt.clf()
        plt.plot(self.reconst_loss, label = "reconst_loss" )
        plt.plot(self.div_loss, label = 'div_loss')
        plt.plot(self.adv_sk, label = 'adv_sk')
        plt.legend()
        plt.savefig('sk_loss.png')
        
        plt.clf()
        plt.plot(self.adv_sd_real, label = 'adv_sd_real')
        plt.plot(self.adv_sd_fake, label = 'adv_sd_fake')
        plt.legend()
        plt.savefig('sd_loss.png')
        

    def save_f_plot(self):
        plt.clf()

        fig = plt.figure()

        counter = 0

        for k,v in self.f.items():
            plt.plot(v, label=k)
            counter+= 1
    

        plt.legend()

        plt.savefig('f_score.png')
       

if __name__ == "__main__":
    Trainer().run()









    
                





        



                


