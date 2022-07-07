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

import Model.resnet as resnet
from Model.fusion import ConCatModule
from Model.base import ImageEncoderTextEncoderBase
from preprocess.loss import NormalizationLayer,BatchHardTripleLoss
#,SequencialMatching,MultiturnNormalizationLayer


class TIRG(ImageEncoderTextEncoderBase):
    """The TIRG model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, args, **kwargs):
        super(TIRG, self).__init__(**kwargs)

        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        normalize_scale = args.normalize_scale
        self.max_turn_len = args.max_turn_len
        self.gru_cell_dim = 1024
        self.model['criterion'] = BatchHardTripleLoss()
        #self.model['criterion'] = SequencialMatching(args=args)
        self.w = nn.Parameter(torch.FloatTensor([1.0, 10.0, 1.0, 1.0]))
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=normalize_scale)
        #self.model['multinorm'] = MultiturnNormalizationLayer(learn_scale=True,
                                        #        normalize_scale=normalize_scale)
        self.linear_layer = nn.Linear(self.gru_cell_dim,self.out_feature_image)
        self.gru = nn.GRU(
            input_size=self.out_feature_image,
            hidden_size=self.gru_cell_dim,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=False,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.model['gated_feature_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )

        self.model['res_info_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )
        self.model = nn.ModuleDict(self.model)

        # optimizer
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

    def compose_img_text(self, imgs, texts):
        image_features = self.extract_image_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_image_text_features(image_features, text_features)

    def compose_image_text_features(self, image_features, text_features):
        f1 = self.model['gated_feature_composer']((image_features, text_features))
        f2 = self.model['res_info_composer']((image_features, text_features))
        f = torch.sigmoid(f1) * image_features * self.w[0] + f2 * self.w[1]
        return f

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if k == 'backbone':
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': float(self.args.lrp)})
            else:
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': 1.0})
        return params

    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr * param_group['lrp']

    def save(self, path, state={}):
        state['state_dict'] = dict()
        for k, v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state, path)

    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        for k, v in state_dict.items():
            self.model[k].load_state_dict(v)

    def get_original_image_feature(self, x):
        '''
        x = image
        '''
        x = self.extract_image_feature(x)
        return self.model['norm'](x)

    def get_manipulated_image_feature(self, x):
        '''
        x[0] = (x_c, c_c, text)
        x[1] = (x_c, c_c, text)
        ...
        x[max_turn_len] = (w_key, )
        '''
        for i in range(self.args.max_turn_len):
            x_f_s = self.compose_img_text(x[i][0], x[i][2])
            x_f_s = torch.unsqueeze(x_f_s,0)
            if (i == 0):
                x_f = x_f_s
            else:
                x_f = torch.cat((x_f, x_f_s), 0)
        candidate_vectors = []
        for pairs in x_f:
            x1 = self.model['norm'](pairs)
            candidate_vectors.append(x1)
        candidate_stack = torch.stack(candidate_vectors,axis=1)
        _,candidate_output = self.gru(candidate_stack,None)
        candidate_output = candidate_output[-1]
        candidate_output = self.model['norm'](self.linear_layer(candidate_output))

        return candidate_output

    def update(self, output, input):

        # assign input
        # loss
        x1 = self.model['norm'](output[0])
        x2 = self.model['norm'](output[1])
        loss = self.model['criterion'](x1, x2)

        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data

    def forward(self, x):
        '''
        data = (
            (x_c, c_c, text),
            (x_c, c_c, text),
            ....
            (x_t, c_t, data['t_id']),
            (we, we_key, text)
        )
        '''
        length = x[self.args.max_turn_len+1][1]
        if self.text_method == 'swem':
            for i in range(self.args.max_turn_len):
                x_f_s = self.compose_img_text(x[i][0], x[2][0])
                if(i==0):
                    x_f = torch.unsqueeze(x_f_s,0)
                else:
                    x_f = torch.cat((x_f,x_f_s),0)

        else:
            for i in range(self.args.max_turn_len):
                x_f_s = self.compose_img_text(x[i][0], x[i][2])
                x_f_s = torch.unsqueeze(x_f_s,0)
                if (i == 0):
                    x_f = x_f_s
                else:
                    x_f = torch.cat((x_f, x_f_s), 0)
        #x_f = self.compose_img_text(x[0][0],x[0][2])+self.compose_img_text(x[1][0],x[1][2])
        #x_f_f = self.model['norm'](torch.mean(x_f,dim=0))
        for i in range(self.args.max_turn_len):
            x_c_s = self.extract_image_feature(x[i][0])
            x_c_s = torch.unsqueeze(x_c_s,0)
            if (i==0):
                x_c = x_c_s
            else:
                x_c = torch.cat((x_c,x_c_s), 0)
        #feed = torch.nn.utils.rnn.pack_padded_sequence(x_f,length,enforce_sorted=False)
        x_t = self.get_original_image_feature(x[self.args.max_turn_len][0])
        # x_f_norm = self.model['multinorm'](x_f)
        # x_t_norm = self.model['norm'](x_t)  # target
        all_output, hidden_ouput = self.gru(x_f, None)
        final_layer = torch.mean(all_output,dim=0)
        #print final_layer.size()
        x_f_f = self.model['norm'](self.linear_layer(final_layer))
        #print x_f_f.data
        #print x_f_t.data
        #print x_f_f.size()

        # if self.text_method == 'swem':
        #     x_f = self.compose_img_text(x[0][0], x[2][0])
        # else:
        #     x_f = self.compose_img_text(x[0][0], x[2][2])
        # x_c = self.extract_image_feature(x[0][0])
        # x_t = self.extract_image_feature(x[1][0])
        return (x_f_f, x_t, x_c)
