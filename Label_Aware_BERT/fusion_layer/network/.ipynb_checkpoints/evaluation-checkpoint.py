
import numpy as np
from scipy import sparse

def get_p_k(predict, target, top):
    prediction = []
    for index_list in predict:
        predicts = [0]*target.shape[1]
        for index in index_list[:top]:
            predicts[index] = 1
        prediction.append(predicts)
    prediction = np.array(prediction)
    target = np.array(target)
    return np.sum(np.multiply(prediction,target))/(top*target.shape[0])


def get_ndcg_k(predict, target, top):

    target = sparse.csr_matrix(np.array(target))
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((target.shape[0], 1))
    
    for i in range(top):
        prediction = []
        for index_list in predict:
            p = index_list[i: i+1]
            predicts = [0]*target.shape[1]
            predicts[p] = 1
            prediction.append(predicts)
        prediction = sparse.csr_matrix(np.array(prediction))
        dcg += prediction.multiply(target).sum(axis=-1) * log[i]
        
    return np.average(dcg / log.cumsum()[np.minimum(target.sum(axis=-1), top) - 1])

