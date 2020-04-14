SD_lr = 0.0002
SK_lr = 0.00001

batch = 1  # this is fixed
epoch = 3

#roots = ['generated_data/summe.h5', 'generated_data/tvsum.h5', 'generated_data/ovp.h5','generated_data/youtube.h5']
#ratios = [0.8, 0.8, 1.0, 1.0]
roots=['generated_data/summe.h5']
ratios = [0.8]
from dataloader import LoadersFactory
factory = LoadersFactory(roots, ratios)