import argparse
import logging
import time
from tqdm import tqdm
import random
import json
import numpy as np


def extract_spans_para(seq):
    para = []
    try:
        task, sents = seq.split('结果是:')
        sent = [s.strip() for s in sents.split('[SSEP]')]
    except:
        task, sent = [], []
    
    if task == 'AOPE':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                at, ot = s.split(' 是 ')

                # if the aspect term is implicit
                if at == '它':
                    at = 'null'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                at, ot = '', ''

            para.append((at, ot))
    
    elif task == 'E2E':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                at, sp = s.split(' 是 ')

                # if the aspect term is implicit
                if at == '它':
                    at = 'null'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                at, sp = '', ''

            para.append((at, sp))
    
    elif task == 'ACSA':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                ac, sp = s.split(' 是 ')
                
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, sp = '', ''

            para.append((ac, sp))
    
    elif task == 'ASTE':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' 因为 ')
                _ , sp = ac_sp.split(' 是 ')
                at, ot  = at_ot.split(' 是 ')

                # if the aspect term is implicit
                if at == '它':
                    at = 'null'
                    
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                at, ot, sp = '', '', ''

            para.append((at, ot, sp))
    
    elif task == 'ACSD':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' 因为 ')
                ac, sp = ac_sp.split(' 是 ')
                at, _  = at_ot.split(' 是 ')

                # if the aspect term is implicit
                if at == '它':
                    at = 'null'

            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                at, ac, sp = '', '', ''

            para.append((at, ac, sp))
    
    elif task == 'ASQP':
    
        for s in sent:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' 因为 ')
                ac, sp = ac_sp.split(' 是 ')
                at, ot = at_ot.split(' 是 ')

                # if the aspect term is implicit
                if at == '它':
                    at = 'null'

            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                at, ot, ac, sp = '', '', '', ''

            para.append((at, ot, ac, sp))
            
    return para


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    # print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return n_gold, n_pred, n_tp


def compute_scores(pred_seqs, gold_seqs):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(gold_seqs[i])
        pred_list = extract_spans_para(pred_seqs[i])

        # print("gold ", gold_seqs[i])
        # print("pred ", pred_seqs[i])

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    #print("all_preds_labels")
    #for i in range(len(all_preds)):
    #    print(all_preds[i], all_labels[i])

    n_gold, n_pred, n_tp = compute_f1_scores(all_preds, all_labels)

    return n_gold, n_pred, n_tp