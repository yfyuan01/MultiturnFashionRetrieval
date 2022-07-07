import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper
import os
import random
from Evaluation import MyIRBench

class Trainer(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.processed_images = 0
        self.global_step = 0

    def __adjust_lr__(self, epoch, warmup=True):
        lr = self.args.lr * self.args.batch_size / 16.0
        if warmup:
            warmup_images = 10000
            lr = min(self.processed_images * lr / float(warmup_images), lr)
        for e in self.args.lr_decay_steps:
            if epoch >= e:
                lr *= self.args.lr_decay_steps
        self.model.adjust_lr(lr)
        self.cur_lr = lr

    def __logging__(self, log_data):
        msg = '[Train][{}]'.format(self.args.expr_name)
        msg += '[Epoch: {}]'.format(self.epoch)
        msg += '[Lr:{:.6f}]'.format(self.cur_lr)
        log_data['lr'] = self.cur_lr
        for k, v in log_data.items():
            if not self.summary_writer is None:
                self.summary_writer.add_scalar(k, v, self.global_step)
            if isinstance(v, float):
                msg += '{}:{:.6f}'.format(k, v)
            else:
                msg += '{}:{}'.format(k, v)
        print msg

    def train(self, epoch):
        self.epoch = epoch
        self.model.train()
        for bidx, input in enumerate(tqdm(self.data_loader, desc='Train')):
            self.global_step += 1
            self.processed_images += input[0][0].size(0)
            self.__adjust_lr__(epoch, warmup=self.args.warmup)
            for i in range(self.args.max_turn_len + 2):
                input[i][0] = Variable(input[i][0]).cuda()
                input[i][1] = Variable(input[i][1]).cuda()
            # input[0][0] = Variable(input[0][0])
            # input[0][1] = Variable(input[0][1])
            # input[1][0] = Variable(input[1][0])
            # input[1][1] = Variable(input[1][1])
            # input[2][0] = Variable(input[2][0])
            # input[self.args.max_turn_len+1] = Variable(input[self.args.max_turn_len+1]).cuda()
            output = self.model(input)
            log_data = self.model.update(output)
            if (bidx % self.args.print_freq) == 0:
                self.__logging__(log_data)


class Evaluator(object):
    def __init__(self, args, data_loader, model, summary_writer, eval_freq):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.eval_freq = eval_freq
        self.best_score = 0
        self.repo_path = os.path.join('./repo', args.expr_name)
        if os.path.exists(self.repo_path):
            os.makedirs(self.repo_path)
        self.targets = list(self.data_loader.keys())

    def test(self, epoch):
        ir_config = {}
        ir_config['srch_method'] = 'bf'
        ir_config['srch_libs'] = None
        ir_config['desc_type'] = 'global'
        irbench = IRBench(ir_config)
        irdict = {}
        self.epoch = epoch
        model = self.model.eval()
        r10 = 0.
        r50 = 0.
        r10r50 = 0.
        for target, data_loader in self.data_loader.items():
            irbench.clean()
            eval_helper = EvalHelper()

            # add index features.
            data_loader.dataset.set_mode('index')
            for bidx, input in enumerate(tqdm(data_loader, desc='Index')):
                input[0] = Variable(input[0].cuda())  # input[0] = (x, image_id)
                data = input[0]
                image_id = input[1]

                with torch.no_grad():
                    output = model.get_original_image_feature(data)
                for i in range(output.size(0)):
                    _iid = image_id[i]
                    _feat = output[i].squeeze().cpu().numpy()
                    irdict[_iid] = _feat
            data_loader.dataset.set_mode('query')
            myirbench = MyIRBench()
            for bidx, input in enumerate(tqdm(data_loader, desc='Query')):
                for i in range(self.args.max_turn_len + 1):
                    input[i][0] = Variable(input[i][0]).cuda()
                    input[i][1] = Variable(input[i][1]).cuda()
                for keys in irdict.keys():
                    input[self.args.max_turn_len + 1][1] = Variable(irdict[keys].repeat(input[i][0].size()[0])).cuda()
                    with torch.no_grad():
                        output = self.model(input)[1]
                        text = self.model(input)[0]
                    for i in range(output.size(0)):
                        _qid = input[self.args.max_turn_len + 1][0][i]
                        _feat = output[i].squeeze().cpu().numpy()
                        _text = text[i].squeeze().cpu().numpy()
                        _tid = input[self.args.max_turn_len][2][i]
                        myirbench.feed_index([_qid, _feat, keys])
                        myirbench.feed_query([_qid, _text])
                        myirbench.feed_gt([_qid, [_tid]])
            myirbench.search_all(2)
            score = myirbench.compute_top_k_acc(kappa=[10, 50])
            print('Target: {}'.format(target))
            print(score)
            _r10 = score['top_10_acc']
            _r50 = score['top_10_acc']
            _r10r50 = 0.5 * (_r10+_r50)
            r10 += _r10
            r50 += _r50
            r10r50 += _r10r50
            if (bidx % self.args.print_freq) == 0 and self.summary_writer is not None:
                self.summary_writer.add_scalar('{}/R10'.format(target), _r10, epoch)
                self.summary_writer.add_scalar('{}/R50'.format(target), _r50, epoch)
                self.summary_writer.add_scalar('{}/R10R50'.format(target), _r10r50, epoch)

            # mean score.
        r10r50 /= len(self.data_loader)
        r10 /= len(self.data_loader)
        r50 /= len(self.data_loader)
        print('Overall>> R10:{:.4f}\tR50:{:.4f}\tR10R50:{:.4f}'.format(r10, r50, r10r50))

