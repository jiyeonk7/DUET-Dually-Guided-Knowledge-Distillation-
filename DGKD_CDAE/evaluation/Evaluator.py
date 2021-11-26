from collections import OrderedDict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor


class Evaluator(object):
    def __init__(self, split_type, topK, num_threads=1):
        self.eval_input = None
        self.eval_target = None
        self.eval_output = None
        self.split_type = split_type
        self.topK = topK if isinstance(topK, list) else [topK]
        self.num_threads = num_threads

    def compute(self, eval_input, eval_target, eval_output):
        self.eval_input = eval_input
        self.eval_target = eval_target
        self.eval_output = eval_output
        score = None    # score: (dictionary) key = 'measure@k', value = score values
        if self.split_type == "loo":
            score = self.compute_by_loo()
        elif self.split_type == "seq_loo":
            score = self.compute_by_loo()
        else:
            print("Please choose a data split type.")
        return score

    def compute_by_loo(self):
        # Find the users in the test dataset.
        #print("topK", self.topK)
        user_idx = []
        for u in range(self.eval_target.shape[0]):
            if len(self.eval_target[u].nonzero()[0]) != 0:
                user_idx.append(u)

        hrs, ndcgs, aps, aucs = {k: [] for k in self.topK}, {k: [] for k in self.topK}, [], []
        if self.num_threads > 1:  # Multi-thread
            with ThreadPoolExecutor() as executor:
                res = executor.map(self.compute_by_loo_user, user_idx)
            res = list(res)
            for _hr, _ndcg, _ap, _auc in res:
                for k in self.topK:
                    hrs[k].append(_hr[k])
                    ndcgs[k].append(_ndcg[k])
                rrs.append(_ap)
                aucs.append(_auc)

        else:   # Single thread
            for u in user_idx:
                (hr, ndcg, ap, auc) = self.compute_by_loo_user(u)
                for k in self.topK:
                    hrs[k].append(hr[k])
                    ndcgs[k].append(ndcg[k])
                aps.append(ap)
                aucs.append(auc)

        score = OrderedDict()

        # hrs = list(map(lambda x: ('HR@%d'%x, np.mean(hrs[x])), hrs))
        # ndcgs = list(map(lambda x: ('NDCG@%d' % x, np.mean(ndcgs[x])), ndcgs))
        hrs = [('HR@%d' % k, np.mean(hrs[k])) for k in hrs]
        ndcg = [('NDCG@%d' % k, np.mean(ndcgs[k])) for k in ndcgs]
        maps = [('MAP', np.mean(aps))]
        auc = [('AUC', np.mean(aucs))]

        score.update(hrs)
        score.update(ndcg)
        score.update(maps)
        score.update(auc)
        return score

    def compute_by_loo_user(self, user_id):
        target_item = self.eval_target[user_id].nonzero()[0][0]
        train_index = self.eval_input[user_id].nonzero()[0]
        predictions = self.eval_output[user_id]
        predictions[train_index] = float('-inf')

        # Sort items by descending oder of scores.
        prediction = predictions.argsort()[::-1]
        hit_at_k = np.where(prediction == target_item)[0] + 1
        #print("hit at k", hit_at_k)

        # hr = 1 if hit_at_k <= self.topK else 0
        # ndcg = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= self.topK else 0
        hr = {k: 1 if hit_at_k <= k else 0 for k in self.topK}
        ndcg = {k: 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0 for k in self.topK}
        ap = 1 / hit_at_k
        #rr = 1 / hit_at_k
        auc = 1 - (1 / 2) * (1 - 1 / hit_at_k)
        return hr, ndcg, ap, auc

    def compute_by_holdout(self):
        # Find the users in the test dataset.
        user_idx = []
        for u in range(self.eval_target.shape[0]):
            if len(self.eval_target[u].nonzero()[0]) != 0:
                user_idx.append(u)

        precs, recalls, ndcgs = {k: [] for k in self.topK}, {k: [] for k in self.topK}, {k: [] for k in self.topK}
        if self.num_threads > 1:  # Multi-thread
            with ThreadPoolExecutor() as executor:
                res = executor.map(self.compute_by_holdout_user, user_idx)
                res = list(res)
                for prec, recall, ndcg in res:
                    for k in self.topK:
                        precs[k].append(prec[k])
                        recalls[k].append(recall[k])
                        ndcgs[k].append(ndcg[k])
        else:
            for u in user_idx:
                (prec, recall, ndcg) = self.compute_by_holdout_user(u)
                for k in self.topK:
                    precs[k].append(prec[k])
                    recalls[k].append(recall[k])
                    ndcgs[k].append(ndcg[k])

        score = OrderedDict()

        # Take mean of each {prec, recall, ndcg} at k
        # result --> prec = [('PREC@5', 0.1234), ('PREC@10', 0.1111)]
        prec = [('PREC@%d' % k, np.mean(precs[k])) for k in precs]
        recall = [('RECALL@%d' % k, np.mean(precs[k])) for k in precs]
        ndcg = [('NDCG@%d' % k, np.mean(precs[k])) for k in precs]

        # prec = list(map(lambda x: ('PREC@%d' % x, np.mean(precs[x])), precs))
        # recall = list(map(lambda x: ('RECALL@%d' % x, np.mean(recalls[x])), recalls))
        # ndcg = list(map(lambda x: ('NDCG@%d' % x, np.mean(ndcgs[x])), ndcgs))

        score.update(prec)
        score.update(recall)
        score.update(ndcg)
        return score

    def compute_by_holdout_user(self, user_id):
        prec, recall, ndcg = {}, {}, {}

        target_item = self.eval_target[user_id].nonzero()[0]
        num_target_items = len(target_item)
        train_index = self.eval_input[user_id].nonzero()[0]
        predictions = self.eval_output[user_id]
        predictions[train_index] = float('-inf')

        # Sort items by descending oder of scores.
        # largest_k = self.topK[-1]
        prediction = predictions.argsort()[::-1]
        # prediction = predictions.argsort()[::-1][:largest_k]
        # hits = np.array([(i + 1, item) for i, item in enumerate(target_item) if item in prediction])
        for k in self.topK:
            pred_k = prediction[:k]
            hits_k = [(i + 1, item) for i, item in enumerate(target_item) if item in pred_k]
            num_hits = len(hits_k)

            idcg_k = 0.0
            for i in range(1, k + 1):
                idcg_k += 1 / math.log(i + 1, 2)

            dcg_k = 0.0
            for idx, item in hits_k:
                dcg_k += 1 / math.log(idx + 1, 2)

            prec[k] = num_hits / k
            recall[k] = num_hits / num_target_items
            ndcg[k] = dcg_k / idcg_k
        return prec, recall, ndcg
