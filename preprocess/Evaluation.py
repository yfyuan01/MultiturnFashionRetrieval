# -*- encoding: utf-8 -*-
import numpy as np
class MyIRBench():
    def __init__(self):
        self.query2id = dict()
        self.id2query = dict()
        self.query_dict = dict()
        self.index_dict = dict()
        self.gt_dict = dict()
        self.rank_dict = dict()
    def feed_index(self, data):
        unique_id = data[0]
        feat = data[1]
        query_id = data[2]
        if query_id not in self.query2id:
            self.query2id[query_id] = len(self.query2id)
            self.id2query[len(self.id2query)] = query_id
        if unique_id not in self.index_dict:
            self.index_dict[unique_id] = []
        self.index_dict[unique_id].append(feat/np.linalg.norm(feat))

    # query 为 text, unique_id 对应query
    def feed_query(self,data):
        unique_id = data[0]
        feat = data[1]
        if unique_id not in self.query_dict:
            self.query_dict[unique_id] = feat/np.linalg.norm(feat)

    def feed_gt(self,data):
        unique_id = data[0]
        query_id_list = data[1]
        self.gt_dict[unique_id] = [self.query2id[q] for q in query_id_list]


    def compute_top_k_acc(self,
            kappa=None):
        """compute top-k retrieval accuracy.
        """
        result = dict()
        for k in kappa:
            rank_dict = self.rank_dict
            gt_dict = self.gt_dict
            hit = 0
            ncnt = 0
            for uid, rank_list in rank_dict.items():
                if not uid in gt_dict:
                    print('no {} key exists in gt_dict'.format(uid))
                    continue;  # if id not exists in GT, skip.
                pos_list = gt_dict[uid]
            # if GT list is empty, skip.
                if len(pos_list) > 0:
                    rank_list = rank_list[:k] if k is not None else rank_list
                # hit if at least one gt exists in rank list.
                    inter_sect = list(set(rank_list).intersection(pos_list))
                    ncnt = ncnt + 1
                    if len(inter_sect) > 0:
                        hit = hit + 1

            top_k_acc = hit / float(ncnt)
            result['top_'+str(k)+'_acc'] = top_k_acc
        return result

    def search_all(self,top_k):
        print('calculating cosine similarity score...')

        for i,query_id in enumerate(self.query_dict.keys()):
            index_feats = np.concatenate(self.index_dict[query_id])
            query_feats = self.query_dict[query_id]
            cosim_s = np.dot(query_feats, index_feats.T)
            if i == 0:
                cosim = cosim_s
            else:
                cosim = np.concatenate((cosim,cosim_s))
        result =  np.argsort(-cosim, axis=1)[:, :top_k] \
            if top_k is not None \
            else np.argsort(-cosim, axis=1)
        for i in range(result.shape[0]):
            self.rank_dict[self.query_dict.keys()[i]] = result[i].tolist()
        return result


if __name__ == '__main__':
    Irbench = MyIRBench()
    index_id = ['target1','target2','target3']
    unique_id = ['candidate_1','candidate_2']
    r = np.array([[[1,2,3],[4,4,4]],[[2,2,2],[4,5,6]],[[0,1,2],[1,2,3]]])

    query_list = [None]*2
    query_list[0] = np.ones([1,3])*2
    query_list[1] = np.ones([1,3])
    for i in range(len(index_id)):
        for j in range(len(unique_id)):
            _feat = r[i][j].reshape(1,3)
            Irbench.feed_index([unique_id[j],_feat,index_id[i]])
    for j in range(len(unique_id)):
        Irbench.feed_query([unique_id[j],query_list[j]])
        Irbench.feed_gt([unique_id[j],[index_id[0]]])

    print Irbench.search_all(2)
    print Irbench.compute_top_k_acc(kappa=[1,2])


