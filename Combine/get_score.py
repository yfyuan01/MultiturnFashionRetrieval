import os
import sys
sys.path.append('../')
import pickle
import time
import random
import json
import easydict
import argparse
from pprint import pprint
from tqdm import tqdm
import numpy as np
from irbench.irbench import IRBench
import torch
from torch.autograd import Variable
from irbench.evals.eval_helper import EvalHelper
from datetime import datetime
from Model.cross_attention import Combine
from Model.TIRG import TIRG
from Model.image_only import ImageOnlyModel
from Model.ComposeAE import ComposeAE
from Model.text_only import TextOnlyModel
from preprocess.dataset_tag import FashionIQTrainValDataset
TOP_K = 50
def init_env():
    state = {k:v for k,v in args._get_kwargs()}
    pprint(state)

    os.environ['CUDA_VISIBLE_DEVIDES'] = args.gpu_id
    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True


def main(args):
    scores = dict()
    hyperopt = dict()
   # print type(score)
    init_env()
    date_key = str(datetime.now().strftime('%Y%m%d%H%M%S'))[2:]
    print 'Load model: {}'.format(args.expr_name)
    #root_path = '../Dialog_Manager'
    #root_path = '../Attr'
    #root_path = '../repo/sum'
    root_path = '/projdata1/info_fil/yfyuan/task202008/Multiturn/repo/combine'
    #root_path = '/projdata17/infofil/yfyuan/task202008/Multiturn/repo/textonly'
    #.format(args.expr_name)
    print root_path
    #with open(os.path.join(root_path, 'args.json'), 'r') as f:
     #   largs = json.load(f)
      #  largs = easydict.EasyDict(largs)
       # pprint(largs)
    image_size = args.image_size
        #texts = torch.load(os.path.join(root_path, 'best_model1_dress_composeae_0.0085.pth'))['texts']
    from preprocess.dataset_tag import FashionIQTrainValDataset
    train_dataset = FashionIQTrainValDataset(
        data_root=args.data_root,
        image_size=image_size,
        image_root=args.image_root,
        split='train',
        target=args.target,
        max_turn_len=args.max_turn_len
                                     )
    texts = train_dataset.get_all_texts()
    if args.method == 'text-only':
        model = TextOnlyModel(args=args,backbone=args.backbone,texts=texts,
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'image-only':
        model = ImageOnlyModel(args=args,backbone=args.backbone,texts=texts,
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                     text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'tirg':
        model = TIRG(args=args,backbone=args.backbone,texts=texts,
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                     text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'composeae':
        model = ComposeAE(args=args,backbone=args.backbone,texts=texts,
    stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                 text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'combine':
        model = Combine(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    #with open(os.path.join(root_path,'args.json'),'r') as f:
     #   largs = json.load(f)
      #  pprint(largs)
       # image_size = largs.image_size
    #model.load(os.path.join(root_path,'best_model1_dress_composeae_0.0085.pth'))
    #model = torch.load(os.path.join(root_path,'best_model_0.029.pkl'))
    model.load_state_dict(torch.load(os.path.join(root_path,'best_model_all_combine_0.181678921614.pkl')))
    model = model.cuda()
    model.eval()
    #print model
    SPLITS = ['val']
    if args.target == 'all':
        targets = ['dress','toptee','shirt']
    else:
        targets = [args.target]
    #targets = ['dress','toptee','shirt']
    #targets = ['dress','toptee','shirt']
    for SPLIT in SPLITS:
        for target in targets:
            print '>> SPLIT: {} / TARGET: {}'.format(SPLIT,target)
            #print type(score)
            scores[target] = dict()
            from preprocess.dataset_tag import FashionIQTestDataset
            test_dataset = FashionIQTestDataset(
                data_root=args.data_root,
                image_size=image_size,
                image_root=args.image_root,
                split='val',
                target=target,
                max_turn_len=args.max_turn_len
            )
            test_loader = test_dataset.get_loader(batch_size=args.batch_size)

            index_ids = []
            index_feats = []
            ir_config = {}
            r10 = 0.
            r50 = 0.
            r10r50 = 0.
            r5 = 0.
            mrr = 0.
            ir_config['srch_method'] = 'bf'
            ir_config['srch_libs'] = None
            ir_config['desc_type'] = 'global'
            irbench = IRBench(ir_config)
            eval_helper = EvalHelper()
            print 'Extract Index Features...'
            test_loader.dataset.set_mode('index')
            for bidx, input in enumerate(tqdm(test_loader,desc='Index')):
                if args.method == 'combine':
                    data = input[2]
                    data1 = Variable(input[0].cuda())
                else:
                    input[0] = Variable(input[0].cuda())
                    data = input[0]
                image_id = input[1]

                with torch.no_grad():
                    if args.method == 'combine':
                        output = model.get_original_combined_feature(data,data1)
                    else:
                        output = model.get_original_image_feature(data)
                for i in range(output.size(0)):
                    _iid = image_id[i]
                    _feat = output[i].squeeze().cpu().numpy()
                    index_feats.append(_feat)
                    irbench.feed_index([_iid, _feat])
                    index_ids.append(_iid)
            index_feats = np.asarray(index_feats)
            query_ids = []
            query_feats = []
            query_dict = {}
            print 'Extract Query Features...'
            test_loader.dataset.set_mode('query')
            for bidx, input in enumerate(tqdm(test_loader)):
                for i in range(args.max_turn_len + 1):
                    input[i][0] = Variable(input[i][0]).cuda()
                    input[i][1] = Variable(input[i][1]).cuda()
                input[args.max_turn_len + 1][1] = Variable(input[args.max_turn_len + 1][1]).cuda()
                with torch.no_grad():
                    output = model(input)[0]
                   # print output.size()
                    # output = model.get_manipulated_image_feature(data)

                for i in range(output.size(0)):
                    _qid = input[args.max_turn_len+1][0][i]
                    _feat = output[i].squeeze().cpu().numpy()
                    query_feats.append(_feat)
                    irbench.feed_query([_qid, _feat])
                    query_ids.append(_qid)
                    _tid = input[args.max_turn_len][2][i]
                    query_dict[_qid] = _tid
                    eval_helper.feed_gt([_qid, [_tid]])
            query_feats = np.asarray(query_feats)
            res = irbench.search_all(top_k=50)
            res = irbench.render_result(res)
            eval_helper.feed_rank_from_dict(res)
            score = eval_helper.evaluate(metric=['top_k_acc'], kappa=[5, 8, 50])
            print('Target: {}'.format(target))
            print(score)
            _r10 = score[0][str(8)]['top_k_acc']
            _r50 = score[0][str(50)]['top_k_acc']
            _r5 = score[0][str(5)]['top_k_acc']
            _r10r50 = 0.5 * (score[0][str(8)]['top_k_acc'] + score[0][str(50)]['top_k_acc'])
            r10 += _r10
            r50 += _r50
            r10r50 += _r10r50
            r5+=_r5

            print 'calculating cosine similarity score...'
            y_score = np.dot(query_feats, index_feats.T)
            y_indices = np.argsort(-1 * y_score,axis=1)
            _hyperopt = {
                'score':y_score,
                'query_ids':query_ids,
                'index_ids':index_ids
            }
            hyperopt[target] = _hyperopt
            _score = []
            _pos = 0.
            for qidx, query_id in enumerate(tqdm(query_ids)):
                _r = []
                pos = list(y_indices[qidx]).index(index_ids.index(query_dict[query_id]))
                _pos+=1.0/(pos+1)
                for j in range(min(TOP_K,y_score.shape[1])):
                    index = y_indices[qidx, j]
                    _r.append([
                        str(index_ids[index]),
                        float(y_score[qidx,index])
                    ])
                _score.append([query_id, _r])
            scores[target] = _score
            _pos=_pos/qidx
            mrr+=_pos
        #mrr /= len(targets)
        #r10r50 /= len(targets)
        #r10 /= len(targets)
        #r50 /= len(targets)
        #r5 /= len(targets)
        print('Overall>> R5:{:.4f}\tR8:{:.4f}\tR50:{:.4f}\tR10R50:{:.4f}\tMRR:{:.4f}'.format(r5,r10,r50,r10r50,mrr))
        print 'Dump top-{} ranking to .pkl file...'.format(TOP_K)
        output_path = './output_score/{}_{}'.format(date_key,args.expr_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join('./output_score/{}_{}'.format(date_key,args.expr_name),'hyperopt.{}.pkl'.format(SPLIT)),'wb') as f:
            pickle.dump(hyperopt, f)
        with open(os.path.join('./output_score/{}_{}'.format(date_key,args.expr_name),'score.{}.pkl'.format(SPLIT)),'wb') as f:
            pickle.dump(scores,f)
        print 'Done.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')

    parser.add_argument('--gpu_id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed', type=int, default=int(time.time()), help='manual seed')
    parser.add_argument('--warmup', action='store_true', help='warmup?')
    parser.add_argument('--expr_name', default='TIRG', type=str, help='experiment name')
    parser.add_argument('--data_root', required=False, type=str, default='../data1/')
    parser.add_argument('--text_method', default='lstm', choices=['lstm', 'swem', 'lstm-gru','encode'], type=str)
    parser.add_argument('--fdims', default=2048, type=int, help='output feature dimensions')
    parser.add_argument('--max_turn_len', default=4)
    parser.add_argument('--method', default='tirg', type=str, help='method')
    parser.add_argument('--target', default='toptee', type=str, help='target (dress | shirt | toptee)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int, help='train batchsize')
    parser.add_argument('--image_size', default=224, type=int, help='image size (default:16)')
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--normalize_scale', default=5.0, type=float)
    parser.add_argument('--lr', default=0.00011148, type=float, help='initial learning rate')
    parser.add_argument('--lrp', default=0.48, type=float, help='lrp')
    parser.add_argument('--lr_decay_factor', default=0.4747, type=float)
    parser.add_argument('--lr_decay_steps', default='10,20,30,40,50,60,70', type=str)
    parser.add_argument('--image_root', type=str, default='/projdata1/info_fil/yfyuan/task202008/CVPR/data/')
    parser.add_argument('--attention_type', type=str, default='dot')
    parser.add_argument('--stack_num', type=int, default=1)
    args,_ = parser.parse_known_args()
    main(args)
