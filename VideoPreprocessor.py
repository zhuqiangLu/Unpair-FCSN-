from FeatureExtractor import FeatureExtractor
import torch.nn as nn
#from kts.cpd_auto import cpd_auto
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
sys.path.insert(0, os.path.join(dir_path, 'kts'))

if __name__ == '__main__':
    from kts.cpd_auto import cpd_auto
    print('es')
#from kts.cpd_auto import cpd_auto


# class Preprocessor():

#     def __init__(self, path, frame_shape=(224, 224), n_frame=320, video_info=None):
#         self.FeatureExtractor = FeatureExtractor()
#         self.frame_shape = frame_shape
#         self.n_frame = n_frame
