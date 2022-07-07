# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys
sys.path.append('../')
import random
import numpy as np
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.datasets as D
import torchvision.models as M
import torchvision.transforms.functional as F
from preprocess.transform import PaddedResize

class FashionIQDataset(data.Dataset):
    def __init__(self,data_root,image_root,max_turn_len,
                 image_size=224,split='val',target='all'):
        self.data_root = data_root
        self.image_root = image_root
        self.target = target
        self.image_size = image_size
        self.max_turn_len = max_turn_len
        self.split = split
        self.transform = None
        self.all_targets = ['dress','toptee','shirt']

        self.reload()
    # image data augmentation
    def __set_transform(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.split in ['test', 'val']:
            self.transform = T.Compose([
            PaddedResize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __load_pil_image__(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB', (224, 224))
            return img

    def __crop_image__(self, img, bbox):
        # left, top, right, buttom
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = x_max + int(h * bbox[3])
        crop_img = img.crop((x_min, y_min, x_max, y_max))
        return crop_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        # print('{} Data Size: {}'.format(self.split,len(self.dataset)))
        print("===============")

    def __load_data__(self):
        raise NotImplementedError()

    def __sample__(self, index):
        raise NotImplementedError()

    def reload(self):
        self.__set_transform()
        self.__load_data__()
        self.__print_status()

    def get_loader(self, **kwargs):
        batch_size = kwargs.get('batch_size', 16)
        num_workers = kwargs.get('workers', 20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True

        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader
# dataset format
# {
# reference_list: [(Image, caption, c_id), (Image, caption, c_id), ...],
# target: (target_url, target_id)
# }
class FashionIQTrainValDataset(FashionIQDataset):
    def __init__(self, **kwargs):
        super(FashionIQTrainValDataset, self).__init__(**kwargs)

    def __load_data__(self):
        with open('/projdata1/info_fil/yfyuan/task202008/CVPR/assets/sentence_embedding/embeddings_2.pkl', 'rb') as f:
            self.we = pickle.load(f)
        with open('/projdata1/info_fil/yfyuan/task202008/CVPR/assets/image_embedding/embeddings_2.pkl', 'rb') as f:
            self.ie = pickle.load(f)

        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        if (self.target == 'all') or (self.target == None):
            targets = self.all_targets
        else:
            targets = [self.target]
        for t in targets:
            cap_file = '{}.{}.json'.format(t,self.split)
            print('[Dataset] load multiturn annotation file: {}'.format(cap_file))
            full_cap_path = os.path.join(self.data_root+'Multiturn/',cap_file)
            print(full_cap_path)
            assert os.path.exists(full_cap_path)
            with open(full_cap_path,'r') as f:
                data = json.load(f)
                for i,d in enumerate(tqdm(data)):
                    ref_list = d['reference']
                    t_id = d['target'][1]
                    reference = []
                    for references in ref_list:
                        c_id = references[2]
                        caption = references[1]
                        if not c_id in self.cls2idx:
                            self.cls2idx[c_id] = len(self.cls2idx)
                            self.idx2cls.append(c_id)
                        self.all_texts.extend(caption)
                        text = [x.strip() for x in caption]
                        random.shuffle(text)
                        text = '[CLS]' + ' [SEP] '.join(text)
                        reference.append((c_id,text))
                    if not t_id in self.cls2idx:
                        self.cls2idx[t_id] = len(self.cls2idx)
                        self.idx2cls.append(t_id)
                    # we_key = '{}_{}_{}_{}'.format(self.split,t,t_id,i)
                    _data = {
                        'reference': reference,
                        't_img_path': os.path.join(self.image_root, 'images/{}.jpg'.format(t_id)),
                        # 'we_key': we_key,
                        't_id': t_id,
                    }
                    self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)
    def __sample__(self,index):
        data = self.dataset[index]
        reference = data['reference']
        reference_list = []
        for ref in reference:
            c_id = ref[0]
            text = ref[1]
            img_path = os.path.join(self.image_root,'images/{}.jpg'.format(c_id))
            x_c = self.__load_pil_image__(img_path)
            c_c = self.cls2idx[c_id]
            reference_list.append([x_c,c_c,text])
        # multiturn padding
        x_t = self.__load_pil_image__(data['t_img_path'])
        c_t = self.cls2idx[data['t_id']]
        # we_key = data['we_key']
        # if we_key in self.we:
        #     we = torch.FloatTensor(self.we[data['we_key']])
        # else:
        #     we = torch.zeros((600))
        t_id = data['t_id']
        if t_id in self.ie:       
            ie = torch.FloatTensor(self.ie[t_id])
        #     # print(ie.size())
        else:
            ie = torch.zeros((2048))
        # print(len(reference_list))
        if not self.transform is None:
            for i in range(len(reference_list)):
                reference_list[i][0] = self.transform(reference_list[i][0])
            x_t = self.transform(x_t)
        if (len(reference) < self.max_turn_len):
            for i in range(self.max_turn_len - len(reference)):
                reference_list.append((torch.zeros([3,self.image_size,self.image_size]), 0, ''))
        for i in range (len(reference_list)):
            reference_list[i] = tuple(reference_list[i])
        reference_list.append((x_t,c_t,data['t_id']))
        reference_list.append((ie,len(reference)))
        return tuple(reference_list)
            # (we,data['we_key']),
            # (ie)
class FashionIQTestDataset(FashionIQDataset):
    def __init__(self, **kwargs):
        super(FashionIQTestDataset, self).__init__(**kwargs)
    def __load_data__(self):
        with open('/projdata1/info_fil/yfyuan/task202008/CVPR/assets/image_embedding/embeddings_2.pkl','rb') as f:
            self.we = pickle.load(f)

        split_file = 'split.{}.{}.json'.format(self.target,self.split)
        print('[Dataset] load split file: {}'.format(split_file))
        self.index_dataset = []
        with open(os.path.join(self.data_root+'image_splits/', split_file), 'rb') as f:
            index_id = json.load(f)
        for id in index_id:
            _data = {
                'img_path': os.path.join(self.image_root, 'images/{}.jpg'.format(id)),
                'img_id': id,
            }
            self.index_dataset.append(_data)
        self.index_dataset = np.asarray(self.index_dataset)
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.query_dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        cap_file = '{}.{}.json'.format(self.target,self.split)
        print('[Dataset] load annotation file: {}'.format(cap_file))
        with open(os.path.join(self.data_root+'Multiturn/',cap_file),'r') as f:
            data = json.load(f)
            for i,d in enumerate(tqdm(data)):
                ref_list = d['reference']
                t_id = d['target'][1]
                first_c_id = d['reference'][0][2]
                reference = []
                for references in ref_list:
                    c_id = references[2]
                    caption = references[1]
                    if not c_id in self.cls2idx:
                        self.cls2idx[c_id] = len(self.cls2idx)
                        self.idx2cls.append(c_id)
                    self.all_texts.extend(caption)
                    text = [x.strip() for x in caption]
                    random.shuffle(text)
                    text = '[CLS]' + ' [SEP] '.join(text)
                    reference.append((c_id, text))
                if not t_id in self.cls2idx:
                    self.cls2idx[t_id] = len(self.cls2idx)
                    self.idx2cls.append(t_id)
                we_key = '{}_{}_{}_{}_{}'.format(self.split,self.target,first_c_id,t_id,i)
                _data = {
                    'reference': reference,
                    't_img_path': os.path.join(self.image_root, 'images/{}.jpg'.format(t_id)),
                    't_id': t_id,
                    'we_key' : we_key
                }
                self.query_dataset.append(_data)
        self.query_dataset = np.asarray(self.query_dataset)

    def set_mode(self, mode):
        assert mode in ['query','index']
        self.mode = mode

    def __len__(self):
        if self.mode == 'query':
            return len(self.query_dataset)
        else:
            return len(self.index_dataset)

    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        print('{} Index Data Size: {}'.format(self.split,len(self.index_dataset)))
        print('{} Query Data Size: {}'.format(self.split,len(self.query_dataset)))
        print("===============")

    def __sample__(self, index):
        if self.mode == 'query':
            return self.__sample_query__(index)
        else:
            return self.__sample_index__(index)
    def __sample_index__(self, index):
        data = self.index_dataset[index]
        x = self.__load_pil_image__(data['img_path'])
        image_id = data['img_id']
        if not self.transform is None:
            x = self.transform(x)
        return (x,image_id)

    def __sample_query__(self, index):
        data = self.query_dataset[index]
        reference = data['reference']
        x_t = self.__load_pil_image__(data['t_img_path'])
        c_t = self.cls2idx[data['t_id']]
        we_key = data['we_key']
        reference_list = []
        for ref in reference:
            c_id = ref[0]
            text = ref[1]
            img_path = os.path.join(self.image_root, 'images/{}.jpg'.format(c_id))
            x_c = self.__load_pil_image__(img_path)
            c_c = self.cls2idx[c_id]
            reference_list.append([x_c, c_c, text])
        if not self.transform is None:
            for i in range(len(reference_list)):
                reference_list[i][0] = self.transform(reference_list[i][0])
            x_t = self.transform(x_t)
        if (len(reference) < self.max_turn_len):
            for i in range(self.max_turn_len - len(reference)):
                reference_list.append((torch.zeros([3,self.image_size,self.image_size]), 0, ''))
        for i in range(len(reference_list)):
            reference_list[i] = tuple(reference_list[i])
        reference_list.append((x_t,c_t,data['t_id']))
        reference_list.append((we_key,len(reference),data['t_id']))
        return tuple(reference_list)

def main():
    train_dataset = FashionIQTrainValDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        image_root=args.image_root,
        split='train',
        target=args.target,
        max_turn_len=args.max_turn_len
    )
    train_loader = train_dataset.get_loader(batch_size=args.batch_size)
    if (args.target == 'all') or (args.target is None):
        targets = ['dress','toptee','shirt']
    else:
        targets = [args.target]
    for bidx,input in enumerate(tqdm(train_loader,desc='Train')):
        continue
    test_dataset = FashionIQTestDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        image_root=args.image_root,
        split='val',
        target=args.target,
        max_turn_len=args.max_turn_len
    )
    test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    test_loader.dataset.set_mode('query')

    for bidx, input in enumerate(tqdm(test_loader, desc='Index')):
        print(input[args.max_turn_len+1][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A Test of this Module')
    parser.add_argument('--data_root',required=False,type=str,default='../data/')
    parser.add_argument('--image_size',default=224,type=int,help='image size (default: 16)')
    parser.add_argument('--target',default='dress',type=str)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--text_method',default='lstm',choices=['lstm','swem','lstm-gru'],
                        type=str)
    parser.add_argument('--method',default='match-text-only',type=str,help='method')
    parser.add_argument('--backbone',default='resnet152',type=str)
    parser.add_argument('--image_root',default='/projdata1/info_fil/yfyuan/task202008/CVPR/data/',type=str)
    parser.add_argument('--max_turn_len',default=4)
    args, _ = parser.parse_known_args()
    main()













