from __future__ import division
from __future__ import print_function

import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# __C.DATASET_NAME = ''  
__C.CONFIG_NAME = ''
__C.DATA_DIR = 'D:\\projects\\t2s-stylegan2-ada\\data\\sketches' 
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.RNN_TYPE = 'LSTM'  # 'GRU'
__C.B_VALIDATION = False

__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 4
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18  # 54

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 15
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = 'D:\\projects\\t2s-stylegan2-ada\\DAMSMencoders\\sketch_18_4_low_da_low_da\\text_encoder.pth'

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64