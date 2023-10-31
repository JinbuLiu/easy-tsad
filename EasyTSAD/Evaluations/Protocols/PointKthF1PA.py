from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import F1class

class PointKthF1PA(EvalInterface):
    def __init__(self, k:int) -> None:
        super().__init__()
        self.eps = 1e-15
        self.k = k
        self.name = "best f1 under %d-delay pa"%self.k
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        k = self.k + margins[0]
            
        search_set = []
        tot_anomaly = 0
        for i in range(labels.shape[0]):
            tot_anomaly += (labels[i] > 0.5)
        flag = 0
        cur_anomaly_len = 0
        cur_max_anomaly_score = 0
        for i in range(labels.shape[0]):
            if labels[i] > 0.5:
                # cur_max_anomaly_score record the highest score in an anomaly segment
                # flag record if entering an anomaly segment
                if flag == 1:
                    cur_anomaly_len += 1
                    if cur_anomaly_len <= k:
                        cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                else:
                    flag = 1
                    cur_anomaly_len = 1
                    cur_max_anomaly_score = scores[i]
            else:
                # reconstruct the score using the highest score
                if flag == 1:
                    flag = 0
                    search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                    search_set.append((scores[i], 1, False))
                else:
                    search_set.append((scores[i], 1, False))
        if flag == 1:
            search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
            
        search_set.sort(key=lambda x: x[0], reverse=True)
        best_f1 = 0
        threshold = 0
        P = 0
        TP = 0
        best_P = 0
        best_TP = 0
        for i in range(len(search_set)):
            P += search_set[i][1]
            if search_set[i][2]:  # for an anomaly point
                TP += search_set[i][1]
            precision = TP / (P + self.eps)
            recall = TP / (tot_anomaly + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            if f1 > best_f1:
                best_f1 = f1
                threshold = search_set[i][0]
                best_P = P
                best_TP = TP
                
        precision = best_TP / (best_P + self.eps)
        recall = best_TP / (tot_anomaly + self.eps)
        return F1class(
            name=self.name, 
            p=float(precision), 
            r=float(recall), 
            f1=float(best_f1), 
            thres=float(threshold)
        )