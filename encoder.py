from __future__ import print_function
# from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from damsm_model import RNN_ENCODER
from training.dataset import prepare_data

import os 
import time
import numpy as np
import sys

class Encoder(object):
    def __init__(self, data_loader, n_words, ixtoword):
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.n_words = n_words
        # self.ixtoword = ixtoword
        self.data_loader = data_loader
        # self.num_batches = len(self.data_loader)

    def sent_encoder_dict(self):
        text_encoder = RNN_ENCODER(self.n_words,
                                       nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                        map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        batch_size = self.batch_size
        cnt = 0
        dict_emb = {}
        for _ in range(1):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 10 == 9:
                    print('step: ', step)
                # if step > 49:
                #     break
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id = prepare_data(
                    data)

                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(
                    captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(
                ), sent_emb.detach()
                for i, key in enumerate(keys):
                    dict_emb[key] = sent_emb[i]
        return dict_emb
# remove words_embs 
    
# TODO: save the dictionary to a file
