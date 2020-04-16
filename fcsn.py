import torch.nn as nn

'''
SK fcsn
'''
class FCSN_ENC_SD(nn.Module):
    def __init__(self):
        super(FCSN_ENC_SD, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv1_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv1_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            #nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/2

        self.conv2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv2_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv2_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            #nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/4

        self.conv3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_3
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            #nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/8

        self.conv4 = nn.Sequential(
            nn.Conv1d(1024, 2048, 3, padding=1),  # conv4_1
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_2
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_3
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            #nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/16

    def forward(self, x):
        h = x
        h = self.conv1(h)
        # print(h.shape)
        h = self.conv2(h)
        # print(h.shape)
        h = self.conv3(h)
        # print(h.shape)
        h = self.conv4(h)
        # print(h.shape)

        return h


class FCSN_MID_SD(nn.Module):
    def __init__(self, n_class):
        super(FCSN_MID_SD, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_1
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_2
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_3
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            #nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/32

        self.conv6 = nn.Sequential(
            nn.Conv1d(2048, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(4096, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.conv8 = nn.Sequential(  # the 1x1 cov layer
            nn.Conv1d(4096, n_class, 3, padding=1),
            nn.BatchNorm1d(n_class),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        h = x
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)
        return h



'''
SK fcsn

'''

class FCSN_ENC(nn.Module):
    def __init__(self):
        super(FCSN_ENC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv1_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv1_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/2

        self.conv2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv2_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv2_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/4

        self.conv3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_1
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_2
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),  # conv3_3
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/8

        self.conv4 = nn.Sequential(
            nn.Conv1d(1024, 2048, 3, padding=1),  # conv4_1
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_2
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_3
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/16

    def forward(self, x):
        h = x
        h = self.conv1(h)
        # print(h.shape)
        h = self.conv2(h)
        # print(h.shape)
        h = self.conv3(h)
        # print(h.shape)
        h = self.conv4(h)
        # print(h.shape)

        return h

class FCSN_MID(nn.Module):
    def __init__(self, n_class):
        super(FCSN_MID, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_1
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_2
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, 3, padding=1),  # conv4_3
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # the temporal max pooling
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        # 1/32

        self.conv6 = nn.Sequential(
            nn.Conv1d(2048, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(4096, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.conv8 = nn.Sequential(  # the 1x1 cov layer
            nn.Conv1d(4096, n_class, 3, padding=1),
            nn.BatchNorm1d(n_class),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        h = x
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)
        return h


class FCSN(nn.Module):

    def __init__(self, n_class=1024):
        super(FCSN, self).__init__()

        self.FCSN_ENC = FCSN_ENC()
        self.FCSN_MID = FCSN_MID(n_class=n_class)
        self.skip = nn.Sequential(
            nn.Conv1d(2048, n_class, 1),
            nn.BatchNorm1d(n_class),
        )
        self.deconv1 = nn.ConvTranspose1d(
            n_class, n_class, 4, padding=1, stride=2, bias=False)
        # the L_out_conv8 = L_out_conv5 as conv6/7/8 use 1x1 kernel
        # L_out_conv5 = (L-1)/2^5 + 1
        # L_out_deconv1 = (L_in -1) * stride - 2 * padding + dilation * (K-1)
        # a skip connection will be established from the output of conv4 to the output of deconv1
        # the setting of deconv1 ensure they have the same length
        self.deconv2 = nn.ConvTranspose1d(
            n_class, n_class, 16, stride=16, bias=False)

    def forward(self, x):
        h = x
        h = self.FCSN_ENC(h)

        skip = self.skip(h)
        h = self.FCSN_MID(h)

        upscore = self.deconv1(h)
        
        h = upscore + skip
        h = self.deconv2(h)

        return h


if __name__ == '__main__':
    import torch
    net = FCSN()
    data = torch.randn((1, 1024, 128))
    print(net(data).shape)
