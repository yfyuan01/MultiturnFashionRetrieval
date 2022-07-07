import torch
import torchvision
import torch.nn as nn
import argparse
from base import ImageEncoderTextEncoderBase
from preprocess.loss import NormalizationLayer,BatchHardTripleLoss
class CrossAttentionModule(nn.Module):
    def __init__(self,args,**kwargs):
        super(CrossAttentionModule, self).__init__(**kwargs)
        self.args = args
        self.linear_layer1 = nn.Sequential(
            nn.Linear(2*self.args.fdims,self.args.fdims),
            nn.Tanh()
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(2 * self.args.fdims,self.args.fdims),
            nn.Tanh())
    # x1 is the output of image-text turns : batch_size * max_turn_len * 2048
    # x2 is the output of target image: batch_size * feature_size * 2048
    def A_To_Q(self, x1, x2):
        h_q = list(x1.size())[1]
        h_a = list(x2.size())[1]
        e = list(x1.size())[2]
        reshape_q = x1.unsqueeze(2)
        reshape_a = x2.unsqueeze(1)
        reshape_q = reshape_q.repeat([1,1,h_a,1])
        reshape_a = reshape_a.repeat([1,h_q,1,1])
        combine = torch.cat([reshape_q,reshape_a],dim=3).reshape([-1,2*e]) # (batch_size*max_turn_len*feature_size)*2e
        M = self.linear_layer1(combine).reshape([-1,h_q,h_a,e]) #(batch_size*max_turn_len*feature_size)*e
        S = torch.softmax(M,dim=1)
        attentive_q = (S*reshape_q).sum(dim=1) #batch_size*feature_size*e
        # similarity = (attentive_q*x2).sum(dim=2) #batch_size*feature_size
        return attentive_q
        # return  similarity

    def Q_To_A(self, x1, x2):
        similarity = self.A_To_Q(x1,x2)
        h_a = list(x2.size())[1]
        e = list(x2.size())[2]
        avg_q = torch.mean(x1,dim=1) #batch_size*2048
        reshape_q = avg_q.unsqueeze(1)
        reshape_q = reshape_q.repeat([1,h_a,1]) #batch_size*feature_size*e
        combine = torch.cat([reshape_q,x2],dim=2).reshape([-1,2*e])
        M = self.linear_layer2(combine).reshape([-1,h_a,e]) #batch_size*feature_size
        S = torch.softmax(M,dim=1)
        attentive_a = S*x2
        return attentive_a
class Combine(ImageEncoderTextEncoderBase):
    def __init__(self,args,**kwargs):
        super(Combine, self).__init__(**kwargs)
        self.args = args
        self.cross_attention = CrossAttentionModule(args=self.args)
        self.gru_cell_dim = 64
        self.model['criterion'] = BatchHardTripleLoss()
        self.gru = nn.GRU(
            input_size=self.out_feature_image,
            hidden_size=self.out_feature_image,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
        )
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=self.args.normalize_scale)
        self.linear_layer = nn.Linear(self.gru_cell_dim,self.out_feature_image)
        self.model = nn.ModuleDict(self.model)
        self.opt = torch.optim.AdamW(
        self.get_config_optim(args.lr),
        lr = args.lr,
        betas = (0.55,0.999)
        )
    def update(self, x):
        x1 = self.model['norm'](x[0])
        x2 = self.model['norm'](x[1])
        loss = self.model['criterion'](x1,x2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data
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
    def get_original_image_feature(self, x):
        x = self.extract_image_feature(x)
        return self.model['norm'](x)
    def get_original_tag_feature(self,x):
        x = self.extract_tag_feature(x).mean(1)
        return self.model['norm'](x)
    def get_original_combined_feature(self,x,y):
        x = self.extract_tag_feature(x)
        y = self.extract_image_feature(y).unsqueeze(1)
        z = torch.cat((x,y),1).mean(1)
        return self.model['norm'](z)


    def forward(self, x):
        for i in range(self.args.max_turn_len):
            x_c_s = self.extract_text_feature(x[i][2]).unsqueeze(1)
            #x_t_s = self.extract_tag_feature(x[i][3])
            #x_i_s = self.extract_image_feature(x[i][0]).unsqueeze(1)
            #x_c_s = torch.cat((x_i_s,x_t_s,x_f_s),1).unsqueeze(0)
            #print x_c_s.size()
            if(i==0):
                x_c = x_c_s
            else:
                x_c = torch.cat((x_c,x_c_s),1)
        #x_tag = x_c.reshape([self.args.max_turn_len,-1,self.args.fdims])
        final_output, _ = self.gru(x_c)
        #final_output = self.linear_layer(torch.mean(all_output, dim=0)).reshape([self.args.batch_size,-1,self.args.fdims])
        #final_output = all_output.reshape([self.args.batch_size,-1,self.args.fdims])
        #print final_output.size()
        x_t = self.extract_tag_feature(x[self.args.max_turn_len][3])
        print x_t.size()
        #x_t_i = self.extract_image_feature(x[self.args.max_turn_len][0]).unsqueeze(1)
        #x_t = torch.cat((x_t_i,x_t_t),1)
        x_1 = self.cross_attention.A_To_Q(x_c,x_t)
        x_2 = self.cross_attention.Q_To_A(x_1,x_t)
        x_1 = self.model['norm'](x_1.mean(1))
        x_2 = self.model['norm'](x_t.mean(1))
        #x_2 = self.get_original_tag_feature(x[self.args.max_turn_len][3])
        return (x_1,x_2)

        #return (final_output,x_t)






def main():
    x1 = torch.randn(3,4,10)
    x2 = torch.randn(3,2,10)
    Co = Combine(args=args)
    CrossAttention = CrossAttentionModule(args=args)
    print CrossAttention.Q_To_A(x1,x2).data
    print Co.cal_attention(x1,x1,x1).size()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('A Test of this Module')
    parser.add_argument('--fdims',default='10',type=int)
    args, _ = parser.parse_known_args()
    main()



