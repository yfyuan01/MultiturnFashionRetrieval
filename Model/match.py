import math
import string
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.autograd import Variable

from TIRG import TIRG
'''
c is the index number of id
input = (
        (x_c,c_c,text),
        (x_c,c_c,text),
        ...,
        (x_c,c_c,text),
        (x_t,c_t,t_id),
        ie
        )
output = (x_f,x_t,x_c)
x_f is the output of fused image and text pair : 4 * 32* 2048
x_t is the output of target image : 32 * 2048
x_c is the output of candidate images only : 4 * 32 * 2048
'''
class MatchBase(object):
    #def __init__(self):
        #super(MatchBase, self).__init__(**kwargs)
       # self.args = args
        #self.max_turn_len = args.max_turn_len
        #self.out_feature_dim = args.fdims
        #self.gru_cell_dim = 64
        #self.linear_layer = nn.Linear(self.gru_cell_dim,self.out_feature_dim)
        #self.rnn = nn.GRUCell(
         #   input_size=self.out_feature_dim,
          #  hidden_size=self.gru_cell_dim,  # rnn hidden unit
           # num_layers=1,  # number of rnn layer
            #batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #)

    def update(self,output,input):
        x2 = self.model['norm'](output[1])
        ie = self.model['norm'](input[self.max_turn_len+1])
        candidate_vetors = []
        for pairs in output[0]:
            x1 = self.model['norm'](pairs)
            candidate_vetors.append(x1)
        candidate_stack = torch.stack(candidate_vetors,axis=1)
        #print candidate_stack.size()
        _,candidate_output = self.gru(candidate_stack,None)
        candidate_output = candidate_output[-1]
        #print candidate_output.size()
        candidate_output = self.model['norm'](self.linear_layer(candidate_output))

        loss = 0
        loss += 1.0-F.cosine_similarity(candidate_output,x2)
        #loss += 1.0-F.cosine_similarity(x2,ie)
        loss /= 2.
        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data


class MatchTIRG(MatchBase,TIRG):
    def __init__(self, **kwargs):
        super(MatchTIRG, self).__init__(**kwargs)













