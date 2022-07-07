import torch
import torch.nn as nn
from Model.base import ImageEncoderTextEncoderBase
from preprocess.loss import (NormalizationLayer,
                             BatchHardTripleLoss)

class TextOnlyModel(ImageEncoderTextEncoderBase):
    def __init__(self,args,**kwargs):
        super(TextOnlyModel,self).__init__(**kwargs)
        self.args = args
        self.max_turn_len = args.max_turn_len
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        self.stack_num = args.stack_num
        self.normalize_scale = args.normalize_scale
        self.gru_cell_dim = 64
        self.model['criterion'] = BatchHardTripleLoss()
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=self.normalize_scale)
        self.model = nn.ModuleDict(self.model)

        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr = args.lr,
            betas = (0.55,0.999)
        )
        self.linear_layer = nn.Linear(self.gru_cell_dim,self.out_feature_image)
        self.gru = nn.GRU(
            input_size=self.out_feature_image,
            hidden_size=self.gru_cell_dim,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=False,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
    def get_config_optim(self,lr):
        params = []
        for k,v in self.model.items():
            if k=='backbone':
                params.append({'params':v.parameters(),'lr':lr*0.1})
            else:
                params.append({'params':v.parameters(),'lr':lr})
        return params

    def adjust_lr(self,lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def save(self,path,state={}):
        state['state_dict'] = dict()
        for k,v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state,path)

    def load(self,path):
        state_dict = torch.load(path)['state_dict']
        for k,v in state_dict.items():
            self.model[k].load_state_dict(v)

    def get_original_image_feature(self,x):
        x = self.extract_image_feature(x)
        return self.model['norm'](x)

    def get_manipulated_image_feature(self,x):
        for i in range(self.args.max_turn_len):
            x_f_s = self.extract_text_feature(x[i][2])
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

    def update(self, output):
        x1 = self.model['norm'](output[0])
        x2 = self.model['norm'](output[1])

        loss = self.model['criterion'](x1,x2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data

    def forward(self, x):
        length = x[self.args.max_turn_len+1][1]
        for i in range(self.args.max_turn_len):
            x_f_s = self.extract_text_feature(x[i][2])
            x_f_s = torch.unsqueeze(x_f_s, 0)
            if (i == 0):
                x_f = x_f_s
            else:
                x_f = torch.cat((x_f, x_f_s), 0)
        for i in range(self.args.max_turn_len):
            x_c_s = self.extract_image_feature(x[i][0])
            x_c_s = torch.unsqueeze(x_c_s, 0)
            if (i == 0):
                x_c = x_c_s
            else:
                x_c = torch.cat((x_c, x_c_s), 0)
        # feed = torch.nn.utils.rnn.pack_padded_sequence(x_f, length, enforce_sorted=False)
        all_output, _ = self.gru(x_f, None)
        final_layer = torch.mean(all_output, dim=0)
        x_f_f = self.model['norm'](self.linear_layer(final_layer))
        x_t = self.extract_image_feature(x[self.args.max_turn_len][0])
        return (x_f_f,x_t,x_c)
