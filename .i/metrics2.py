from __future__ import division

import numpy as np

from utils import group_offsets


class Scorer(object):
    def __init__(self, score_func, **kwargs):
        self.score_func = score_func
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.score_func(*args, **self.kwargs)


# DCG/nDCG (Normalized Discounted Cumulative Gain)
#
def _burges_dcg(y_true, y_pred, k=None):
    order = np.argsort(-y_pred)
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(gain)) + 2)
    return np.sum(gain / discounts)

def _burges_dcg1(y_true, y_pred, k=None):
    print("Init")
    print("y_true",y_true)
    print("y_pred",y_pred)
    order = np.argsort(-y_pred)
    print("order",order)
    y_true = np.take(y_true, order[:k])
    print("y_true",y_true)
    gain = 2 ** y_true - 1
    print("gain",gain)
    discounts = np.log(np.arange(len(gain)) + 2)
    print("discounts",discounts)
    print("",np.sum(gain / discounts))
    return np.sum(gain / discounts)



def _dcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
    assert dcg_func is not None
    y_true = np.maximum(y_true, 0)
    a = np.array([dcg_func(y_true[a:b], y_pred[a:b], k=k) for a, b in group_offsets(qid)])
    return a


def _ndcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
    assert dcg_func is not None
    y_true = np.maximum(y_true, 0)
    dcg = _dcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)

    idcg = np.array([_burges_dcg1(np.sort(y_true[a:b]), np.arange(0, b - a), k=k)
                     for a, b in group_offsets(qid)])
    # ACA
    print(dcg, idcg)
    assert (dcg <= idcg).all()
    idcg[idcg == 0] = 1
    return dcg / idcg


def ndcg_score(y_true, y_pred, qid, k=None, version='burges'):
    assert version in ['burges', 'trec']
    dcg_func = _burges_dcg
    return _ndcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)

class NDCGScorer(Scorer):
    def __init__(self, **kwargs):
        super(NDCGScorer, self).__init__(ndcg_score, **kwargs)