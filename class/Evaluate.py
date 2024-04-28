import math
import numpy as np
from util import find_k_largest


class Evaluate():
    def __init__(self, conf):
        self.conf = conf

    def getndcg(self, predict, positive):
        DCG, IDCG = 0, 0
        length = min(len(predict), len(positive))
        for i, item in enumerate(predict):
            if item in positive:
                DCG += math.log(2) / math.log(i + 2)
        for i in range(length):
            IDCG += math.log(2) / math.log(i + 2)
        ndcg = DCG / IDCG
        return ndcg

    def evaluateRankingPerformance_fullsort(self, eva_prediction, rated_data, true_data, topK):
        user_list = list(true_data.keys())
        recall_list, ndcg_list = [], []
        Rec_list = {}
        for u in user_list:
            u_predict = eva_prediction[u]
            u_rated_item = rated_data[u]
            u_true_item = true_data[u]
            positive_length = len(u_true_item)
            target_length = min(positive_length, topK)
            u_predict[list(u_rated_item)] = -1<<10
            topk_item, _ = find_k_largest(topK, u_predict)
            Rec_list[u] = topk_item
            hits = len(set(u_true_item).intersection(set(topk_item)))
            tmp_recall = hits / target_length
            tmp_ndcg = self.getndcg(topk_item, u_true_item)
            recall_list.append(tmp_recall)
            ndcg_list.append(tmp_ndcg)
        recall, ndcg = np.mean(recall_list), np.mean(ndcg_list)
        return recall, ndcg, Rec_list


    def rankingMeasure(self, Recdict, true_data, item_user):
        self.item_user = item_user
        recall_list, ndcg_list = [], []
        recitem = list()
        item_count = dict()
        user_list = list(true_data.keys())
        for u in user_list:
            predict_item_u = Recdict[u]
            recitem.extend(list(predict_item_u))
            for item in predict_item_u:
                item_count[item] = item_count.get(item, 0) + 1
            positive_item_u = true_data[u]
            hits = len(positive_item_u.intersection(set(predict_item_u)))
            u_recall = hits / min(len(positive_item_u), self.conf.topk)
            DCG, IDCG = 0, 0
            length = min(len(predict_item_u), len(positive_item_u))
            for i, item in enumerate(predict_item_u):
                if item in positive_item_u:
                    DCG += math.log(2) / math.log(i + 2)
            for i in range(length):
                IDCG += math.log(2) / math.log(i + 2)
            u_ndcg = DCG / IDCG
            recall_list.append(u_recall)
            ndcg_list.append(u_ndcg)

        recall, ndcg = np.mean(recall_list), np.mean(ndcg_list)
        aggdiv = len(set(recitem)) / self.conf.num_items
        item_pi = dict()
        Num = sum(item_count.values())
        for k, v in item_count.items():
            item_pi[k] = v / Num
        entory = sum([pi * (-math.log(pi) / math.log(2)) for i, pi in item_pi.items()])
        measure = [recall, ndcg, aggdiv, entory]
        return measure