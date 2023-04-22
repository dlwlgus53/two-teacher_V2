import os
import pdb
import pickle
import ontology
import csv, json
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score



def acc_metric(gold, pred, type):
    score_dict = {
        'acc' : accuracy_score(gold, pred)
    }
    for type_ in list(set(type)):
        gold_ = [g for g, t in zip(gold, type) if t == type_]
        pred_ = [p for p, t in zip(pred, type) if t == type_]
        score_dict[type_] = accuracy_score(gold_, pred_)
    return score_dict





def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
    
            
def compute_acc(gold, pred, slot_temp):
    detail_wrong = []
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.split(" : ")[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def cal_num_same(a,p):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    return num_same


def cal_f1(a, p, return_all = False):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        precision = 0
        recall =0
        mini_f1 = 0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(answer_tokens)
        mini_f1 = (2 * precision * recall) / (precision + recall)
    if return_all:
        return precision, recall, mini_f1
    else:
        return mini_f1


if __name__ == '__main__':
    seed = 2
    # pred_file = json.load(open(f'logs/baseline_sample{seed}/pred_belief.json', "r"))
    pred_file = json.load(open(f'logs/baseline_sample1.0/pred_belief.json', "r"))

    ans_file = json.load(open('../woz-data/MultiWOZ_2.1/test_data.json' , "r"))
    # unseen_data = json.load(open(f'../woz-data/MultiWOZ_2.1/split0.1/unseen_data{seed}0.1.json' , "r"))
    unseen_data = json.load(open(f'../woz-data/MultiWOZ_2.1/unseen_data.json' , "r"))

    JGA, slot_acc, unseen_recall = evaluate_metrics(pred_file, ans_file, unseen_data)
    print(f'JGA : {JGA*100:.4f} ,  slot_acc : {slot_acc*100:.4f}, unseen_recall : {unseen_recall*100:.4f}')

