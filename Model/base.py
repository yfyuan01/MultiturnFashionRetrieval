import os
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.autograd import Variable
import pickle
import urllib2
from tqdm import tqdm
import resnet as resnet
from preprocess.spellchecker import SpellChecker
import string
from preprocess.loss import MultiturnNormalizationLayer
__VERBOSE__ = False

class SimpleVocab(object):
    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.spell = SpellChecker()
        self.word2id = {}
        self.wordcount = {}
        self.add_special_token('[PAD]')
        self.add_special_token('[CLS]')
        self.add_special_token('[SEP]')
    def add_special_token(self,token):
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 9e9

    def token_text(self,text):
        text = text.encode('ascii','ignore').decode('ascii')
        table = string.maketrans('','')
        tokens = str(text).lower().translate(table,string.punctuation).strip().split()
        return tokens

    def add_text_to_vocab(self,text):
        tokens = self.token_text(text)
        if __VERBOSE__:
            print '[Tokenizer] Text: {}/Tokens:{}'.format(text,tokens)
        for token in tokens:
            token = SpellChecker.correct_token(token)
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def encode_text(self,text):
        tokens = self.token_text(text)
        if len(tokens)>0:
            x = [self.word2id.get(SpellChecker.correct_token(token),0) for token in tokens]
        else:
            x = [0]
        return x

    def get_size(self):
        return len(self.word2id)

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask:
        	attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=900, ffn_dim=900, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=900, ffn_dim=900, dropout=0.0, stack_num=3):
        super(EncoderLayer, self).__init__()
        self.stack_num = stack_num
        self.model_dim=model_dim
        self.ffn_dim=ffn_dim
        self.attention = ScaledDotProductAttention(dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    def forward(self, inputs, attn_mask=None):
        # self attention
        for i in range(self.stack_num):
            context, attention = self.attention(inputs, inputs, inputs)
        # feed forward network
            inputs = self.feed_forward(context)
        return inputs

class Word2Vec(nn.Module):
    def __init__(self,vocab,embed_size,init_with_glove=False):
        super(Word2Vec, self).__init__()
        vocab_size = vocab.get_size()
        self.embedding = nn.Embedding(vocab_size,embed_size)

        if init_with_glove:
            print "Initialize Word2Vec with GloVe vectors..."
            word2vec_path = 'assets/word2vec'
            with open(os.path.join(word2vec_path,'glove_840B_300d/embeddings_2.pkl'),'rb') as f:
                self.word2vec_a = pickle.load(f)
            with open(os.path.join(word2vec_path,'glove_42B_300d/embeddings_2.pkl'),'rb') as f:
                self.word2vec_b = pickle.load(f)
            with open(os.path.join(word2vec_path,'glove_6B_300d/embeddings_2.pkl'),'rb') as f:
                self.word2vec_c = pickle.load(f)
            print "Done."

            word2id = vocab.word2id
            id2word = [0]*len(word2id)
            for k,v in word2id.items():
                id2word[v] = k
            weights = []
            for idx,word in enumerate(tqdm(id2word)):
                if (word in self.word2vec_a) and (word in self.word2vec_b) and (word in self.word2vec_c):
                    w_a = torch.FloatTensor(self.word2vec_a[word])
                    w_b = torch.FloatTensor(self.word2vec_b[word])
                    w_c = torch.FloatTensor(self.word2vec_c[word])
                    w = torch.cat([w_a,w_b,w_c])
                    weights.append(w)
                else:
                    if __VERBOSE__:
                        print '[Warn] token {} does not exist in GloVe'.format(word)
                    weights.append(self.embedding.weight[idx])
        #   convert to a tensor
            weights = torch.stack(weights)
            self.embedding.weight = nn.Parameter(weights)

    def forward(self, x):
        return self.embedding(x)

class TextSWEMModel(nn.Module):
    def __init__(self,
                 fc_arch,
                 in_dim,
                 out_dim):
        super(TextSWEMModel,self).__init__()
        if fc_arch == 'A':
            self.fc_output = torch.nn.Sequential(
                torch.nn.BatchNorm1d(in_dim),
                torch.nn.Linear(in_dim,out_dim),
            )
        elif fc_arch == 'B':
            self.fc_output = nn.Linear(in_dim,out_dim)
    def forward(self, x):
        return self.fc_output(x)

class TextLSTMModel(nn.Module):
    def __init__(self,
                 fc_arch,
                 texts_to_build_vocab,
                 word_embed_dim,
                 lstm_hidden_dim,
                 init_with_glove):
        super(TextLSTMModel,self).__init__()
        self.vocab = SimpleVocab()
        for text in tqdm(texts_to_build_vocab):
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.lstm_hiddem_dim = lstm_hidden_dim
        self.embedding_layer = Word2Vec(self.vocab,word_embed_dim,init_with_glove=init_with_glove)

        self.num_layers = 2
        self.lstm = torch.nn.LSTM(word_embed_dim,lstm_hidden_dim,num_layers=self.num_layers,dropout=0.1)
        self.encoder = EncoderLayer(dropout=0.1)

        if fc_arch=='A':
            self.fc_output = torch.nn.Sequential(torch.nn.BatchNorm1d(lstm_hidden_dim),
                                                 torch.nn.Linear(lstm_hidden_dim,2048))
                                                # lstm_hidden_dim))
        elif fc_arch=='B':
            self.fc_output = nn.Linear(lstm_hidden_dim,lstm_hidden_dim)

    def forward(self, x):
        if isinstance(x,list) or isinstance(x,tuple):
            if isinstance(x[0],str) or isinstance(x[0],unicode):
                x = [self.vocab.encode_text(text) for text in x]
        assert isinstance(x,list) or isinstance(x,tuple)
        assert isinstance(x[0],list) or isinstance(x[0],tuple)
        assert isinstance(x[0][0],int)
        return self.forward_encoded_texts(x)


    def forward_encoded_texts(self,texts):
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths),len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i],i] = torch.LongTensor(texts[i])
        #itexts = Variable(itexts)
        itexts = Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)
        lstm_output,_ = self.forward_lstm_(etexts)
        text_features = []
        for i in range(len(texts)):
           text_features.append(lstm_output[lengths[i]-1,i,:])
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features

    def forward_lstm_(self,etexts):
        batch_size = etexts.shape[1]
        first_hidden = (
            #torch.zeros(self.num_layers, batch_size, self.lstm_hiddem_dim),
            #torch.zeros(self.num_layers, batch_size, self.lstm_hiddem_dim),
            torch.zeros(self.num_layers,batch_size,self.lstm_hiddem_dim).cuda(),
            torch.zeros(self.num_layers,batch_size,self.lstm_hiddem_dim).cuda(),
        )
        # etexts : sequence len * batch size * embedding len
        lstmoutput, last_hidden = self.lstm(etexts,first_hidden)
        return lstmoutput,last_hidden
class TextSelfAttentionModel(nn.Module):
    def __init__(self,
                 fc_arch,
                 texts_to_build_vocab,
                 word_embed_dim,
                 hidden_dim,
                 init_with_glove,
                 stack_num,
                 ):
        super(TextSelfAttentionModel,self).__init__()
        self.vocab = SimpleVocab()
        for text in tqdm(texts_to_build_vocab):
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()
        self.out_dim = hidden_dim
        self.word_embed_dim = word_embed_dim
        self.embedding_layer = Word2Vec(self.vocab, word_embed_dim, init_with_glove=init_with_glove)
        self.encoder = EncoderLayer(dropout=0.1,stack_num=stack_num)

        if fc_arch == 'A':
            self.fc_output = torch.nn.Sequential(torch.nn.BatchNorm1d(word_embed_dim),
                                                 torch.nn.Linear(word_embed_dim, hidden_dim))
            # lstm_hidden_dim))
        elif fc_arch == 'B':
            self.fc_output = nn.Linear(word_embed_dim, hidden_dim)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            if isinstance(x[0], str) or isinstance(x[0], unicode):
                x = [self.vocab.encode_text(text) for text in x]
        assert isinstance(x, list) or isinstance(x, tuple)
        assert isinstance(x[0], list) or isinstance(x[0], tuple)
        assert isinstance(x[0][0], int)
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i], i] = torch.LongTensor(texts[i])
        # itexts = Variable(itexts)
        itexts = Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)
        attention_output = self.encoder(etexts.transpose(0, 1)).transpose(0, 1)
        text_features = attention_output.mean(0)
        text_features = self.fc_output(text_features)
        return text_features


class TextLSTMGRUModel(nn.Module):

    def __init__(self,
                 fc_arch,
                 texts_to_build_vocab,
                 word_embed_dim,
                 hidden_dim,
                 init_with_glove):
        super(TextLSTMGRUModel,self).__init__()

        self.vocab = SimpleVocab()
        for text in tqdm(texts_to_build_vocab):
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim
        self.embedding_layer = Word2Vec(self.vocab,word_embed_dim,init_with_glove=init_with_glove)

        self.num_layers = 2
        self.lstm = torch.nn.LSTM(word_embed_dim,hidden_dim,
                                  num_layers=self.num_layers,
                                  dropout=0.1,
                                  bidirectional=False)

        self.num_layers = 2
        self.gru = torch.nn.GRU(word_embed_dim,hidden_dim,
                                num_layers=self.num_layers,
                                dropout=0.1,
                                bidirectional=False)
        if fc_arch=='A':
            self.fc_output = torch.nn.Sequential(
                nn.BatchNorm1d(2*hidden_dim),
                nn.Linear(2*hidden_dim,hidden_dim),
            )
        elif fc_arch == 'B':
            self.fc_output = nn.Linear(2*hidden_dim,hidden_dim)
    def forward(self, x):
        if isinstance(x,list) or isinstance(x,tuple):
            if isinstance(x[0],str) or isinstance(x[0],unicode):
                x = [self.vocab.encode_text(text) for text in x]
        assert isinstance(x,list) or isinstance(x,tuple)
        assert isinstance(x[0],list) or isinstance(x[0],tuple)
        assert isinstance(x[0][0],int)
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self,texts):
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths),len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i],i] = torch.LongTensor(texts[i])
        #itexts = Variable(itexts)
        itexts = Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        lstm_output,_ = self.forward_lstm_(etexts)
        gru_output,_ = self.forward_gru_(etexts)

        text_features = []
        for i in range(len(texts)):
            _lstm_feat = lstm_output[lengths[i]-1,i,:]
            _gru_feat = gru_output[lengths[i]-1,i,:]
            text_features.append(torch.cat([_lstm_feat,_gru_feat]))
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features

    def forward_lstm_(self,etexts):
        lstm_output,last_hidden = self.lstm(etexts)
        return lstm_output,last_hidden
    def forward_gru_(self,etexts):
        gru_output,last_hidden = self.gru(etexts)
        return gru_output,last_hidden

class ImageEncoderTextEncoderBase(nn.Module):
    def __init__(self,backbone,texts,text_method,fdims,stack_num,max_turn_len,normalize_scale,init_with_glove,fc_arch):
        super(ImageEncoderTextEncoderBase, self).__init__()
        self.in_feature_image = None
        self.out_feature_image = fdims
        self.in_feature_text = 3*300
        self.out_feature_text = fdims
        self.stack_num = stack_num
        self.max_turn_len = max_turn_len
        self.normalize_scale = normalize_scale

        pretrained = True
        if backbone in [
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152', 'resnext50_32*4d', 'resnext101_32*8d',
            'wide_resnet50_2','wide_resnet101_2'
        ]:
            self.backbone = resnet.__dict__[backbone](pretrained=pretrained,num_classes=1000)
        print "Backbone: {} is loaded with pretrained={}".format(backbone,pretrained)
        if backbone in ['resnet18','resnet34']:
            self.in_feature_image = 512
        else:
            self.in_feature_image = 2048

        if init_with_glove:
            self.in_feature_text = 3*300
        self.model = dict()
        self.model['backbone'] = self.backbone
        self.model['image_encoder'] = nn.Sequential(
            nn.Linear(self.in_feature_image,self.out_feature_image)
        )
        if text_method =='lstm':
            self.model['text_encoder'] = TextLSTMModel(
                fc_arch=fc_arch,
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                lstm_hidden_dim=self.out_feature_text,
                #self.out_feature_text,
                init_with_glove=init_with_glove,
            )
        elif text_method == 'lstm-gru':
            self.model['text_encoder'] = TextLSTMGRUModel(
                fc_arch=fc_arch,
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                hidden_dim=self.out_feature_text,
                init_with_glove=init_with_glove,
            )
        elif text_method == 'swem':
            self.in_feature_text = 600
            self.out_feature_text = 600
            self.model['text_encoder'] = TextSWEMModel(
                fc_arch=fc_arch,
                in_dim=self.in_feature_text,
                out_dim=self.out_feature_text,
            )
        elif text_method == 'encode':
            self.model['text_encoder'] = TextSelfAttentionModel(
                fc_arch=fc_arch,
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                hidden_dim=self.out_feature_text,
                init_with_glove=init_with_glove,
                stack_num=stack_num
            )
    def extract_image_feature(self,x):
        x = self.model['backbone'](x)
        x = self.model['image_encoder'](x)
        return x
    def extract_text_feature(self,texts):
        x = self.model['text_encoder'](texts)
        return x
    def extract_tag_feature(self,tag):
        for i in range(len(tag)):
            x = self.model['text_encoder'](tag[i]).unsqueeze(1)
            if i == 0:
                x_f = x
            else:
                x_f = torch.cat((x_f,x),dim=1)
        return x_f





