import argparse
import logging
import time
from tqdm import tqdm
import random
import json
import numpy as np

class Span_Evalution:
    def __init__(self, outputs, targets):
        
        assert len(outputs) == len(targets)
        self.predict_quads = []
        self.gold_quads = []
        self.outputs = outputs
        self.targets = targets
        for i in range(len(targets)):
            self.predict_quads.append(extract_spans_para(outputs[i]))
            self.gold_quads.append(extract_spans_para(targets[i]))
        
        
    def jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union if union !=0 else 0
    
    def Quad_Score(self, pred, gold):
        if len(pred) != 4:
            score1, score2, score3, score4 =0, 0, 0, 0
        else:
            A_T_P, O_T_P = pred[0], pred[1] 
            A_T_G, O_T_G = gold[0], gold[1]
            A_T_P = [char for char in A_T_P]
            O_T_P = [char for char in O_T_P]
            A_T_G = [char for char in A_T_G]
            O_T_G = [char for char in O_T_G]
            score1 = self.jaccard_similarity(A_T_P, A_T_G)
            score2 = self.jaccard_similarity(O_T_P, O_T_G)
            score3 = 1 if pred[2] == gold[2] else 0
            score4 = 1 if pred[3] == gold[3] else 0

        return (score1, score2, score3, score4)
    
    def list_Score(self, predict_list, gold_list):
        score = 0
        for ele in predict_list:
            temp= []
            for ele2 in gold_list:
                temp.append(sum(self.Quad_Score(ele, ele2))/4)
            max_score = max(temp)
            score += max_score
        
        return (score/len(predict_list)) if len(predict_list) != 0 else 0
    
    def Quad_Evaluation(self):
        final_score = 0
        for i in range(len(self.predict_quads)):
            score = self.list_Score(self.predict_quads[i], self.gold_quads[i])
            final_score += score

        return final_score/len(self.predict_quads)