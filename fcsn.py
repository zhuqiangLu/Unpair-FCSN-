import torch.nn as nn


class FCSN(nn.Module):

    '''
    the input shape should be (n * c * l)
    n: the number of tensor
    c: the number of channel or the dim of feature vector
    l: the length
    '''
    def __init__(self, n_class=2):
        super(FCSN, self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1), #conv1_1
            nn.BatchNorm1d(1024),
            nn.ReLu(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1), #conv1_2
            nn.BatchNorm1d(1024),
            nn.ReLu(inplace=True),
            nn.MaxPool1D(2, 2, ceil_mode=True),  #the temporal max pooling
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/2

        self.conv2=nn.Sequential(
            nn.Conv1d(1024, 1024, 3, padding=1), #conv2_1
            nn.BatchNorm1d(1024),
            nn.ReLu(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1), #conv2_2
            nn.BatchNorm1d(1024),
            nn.ReLu(inplace=True),
            nn.MaxPool1D(2, 2, ceil_mode=True),  #the temporal max pooling
        )

        # the length after conv1 will be ceiling((l-1)/2 + 1) near 1/4

        self.conv3=nn.




