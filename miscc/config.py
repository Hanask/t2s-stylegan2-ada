from __future__ import division
from __future__ import print_function

import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET_NAME = '' # TO DO 
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''  # TO DO
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
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''