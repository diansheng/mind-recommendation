#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score

"""
Reference:
1. https://github.com/microsoft/recommenders/blob/master/examples/00_quick_start/nrms_MIND.ipynb
2. https://github.com/microsoft/recommenders/blob/711d58499dee565ef1708d7f358774f124a63e1f/reco_utils/recommender/deeprec/deeprec_utils.py#L515
2. https://github.com/msnews/MIND/blob/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py#L27
"""


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks


def scoring_from_relevence(truth, pred):
    """
    truth: list of list. the inner list is a list of 0 or 1, indicating whether certain news is clicked
    pred: same structure as truth. instead of 0 or 1s, score between 0 and 1.
    """
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    for y_true, y_score in zip(truth, pred):
        auc = roc_auc_score(y_true,y_score)
        mrr = mrr_score(y_true,y_score)
        ndcg5 = ndcg_score(y_true,y_score,5)
        ndcg10 = ndcg_score(y_true,y_score,10)
        
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
    
    auc, mrr, ndcg, ndcg10 = np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
    print("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))

    return auc, mrr, ndcg, ndcg10
        

def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        
        # ignore masked impressions
        if labels == []:
            continue 
        
        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))       
        
        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))        
        
        lt_len = float(len(labels))
        
        y_true =  np.array(labels,dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1./rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)
            
        auc = roc_auc_score(y_true,y_score)
        mrr = mrr_score(y_true,y_score)
        ndcg5 = ndcg_score(y_true,y_score,5)
        ndcg10 = ndcg_score(y_true,y_score,10)
        
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        
        line_index += 1

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
        

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res') 
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')              
        output_file = open(output_filename, 'w')

        truth_file = open(os.path.join(truth_dir, "truth.txt"), 'r')
        submission_answer_file = open(os.path.join(submit_dir, "prediction.txt"), 'r')
        
        auc, mrr, ndcg, ndcg10 = scoring(truth_file, submission_answer_file)

        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
        output_file.close()