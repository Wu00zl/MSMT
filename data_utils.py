import argparse
import logging
import time
from tqdm import tqdm
import random
import json
import numpy as np
from torch.backends import cudnn
from torch.utils.data import Dataset

senttag2word = {'POS': '正', 'NEG': '负', 'NEU': '中'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'正': '好', '负': '差', '中': '一般'}
aspect_cate_list = ['售后', '包装', '色泽', '价格', '品质', '口感', '物流', '分量', '其他']

def read_line_examples_from_file(data_path, silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r') as f:
        comments = json.load(f)
        
    for comment in comments:
        sents.append(comment['comments'])
        labels.append(comment['label'])
        
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels

def get_para_targets(labels, task):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    AOPE_Pre = 'AOPE结果是：'
    E2E_Pre  = 'E2E结果是：'
    ACSA_Pre = 'ACSA结果是：'
    ASTE_Pre = 'ASTE结果是：'
    ACSD_Pre = 'ACSD结果是：'
    ASQP_Pre = 'ASQP结果是：'
    
    if task == 'AOPE':
        _prompt = '（方面术语；观点术语）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                if len(quad) == 4:
                    at, ot, ac, sp = quad
                elif len(quad) == 3:
                    at, ot, sp = quad

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{at} 是 {ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = AOPE_Pre + target
            targets.append(target)
            
    elif task == 'E2E':
        _prompt = '（方面术语；情感极性）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                if len(quad) == 4:
                    at, ot, ac, sp = quad
                elif len(quad) == 3:
                    at, ot, sp = quad

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{at} 是 {man_ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = E2E_Pre + target
            targets.append(target)
    
    elif task == 'ACSA':
        _prompt = '（方面类别；情感极性）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                at, ot, ac, sp = quad

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{ac} 是 {man_ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = ACSA_Pre + target
            targets.append(target)
    
    elif task == 'ASTE':
        _prompt = '（方面术语；观点术语；情感极性）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                if len(quad) == 4:
                    at, ot, ac, sp = quad
                elif len(quad) == 3:
                    at, ot, sp = quad

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{at} 是 {man_ot} 因为 {at} 是 {ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = ASTE_Pre + target
            targets.append(target)
    
    elif task == 'ACSD':
        _prompt = '（方面术语；方面类别；情感极性）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                at, ot, ac, sp = quad

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{ac} 是 {man_ot} 因为 {at} 是 {man_ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = ACSD_Pre + target
            targets.append(target)
    
    elif task == 'ASQP':
        _prompt = '（方面术语；观点术语；方面类别；情感极性）'
        
        for label in labels:
            all_sentences = []
            for quad in label:
                at, ot, ac, sp = quad

                if at == 'null':  # for implicit aspect termACSP
                    at = '它'

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                one_sentence = f"{ac} 是 {man_ot} 因为 {at} 是 {ot}"
                all_sentences.append(one_sentence)
            target = ' [SSEP] '.join(all_sentences)
            target = ASQP_Pre + target
            targets.append(target)

    return targets, _prompt

def get_transformed_io(sents, labels, task):
    
    targets, _prompt = get_para_targets(labels, task)
    
    task_pre = task + _prompt +'任务抽取：'
    
    new_sents = []
    for sent in sents:
        new_sents.append(task_pre + sent)

    return new_sents, targets


class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 args,
                 input_max_len = 200,
                 target_max_len = 300):
        self.tokenizer = tokenizer
        self.max_input_max = input_max_len
        self.max_target_len = target_max_len
        self.args = args

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):
        
        max_input_len, max_target_len = 0, 0
        if self.args.train == True:
            inputs, targets = [], []
            JD_sents, JD_targets = read_line_examples_from_file(self.args.train_dataset)
            MT_sents, MT_targets = read_line_examples_from_file(self.args.meituan_train_dataset)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'ASQP')
            inputs.extend(_sents)
            targets.extend(_targets)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'ACSD')
            inputs.extend(_sents)
            targets.extend(_targets)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'ACSA')
            inputs.extend(_sents)
            targets.extend(_targets)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'ASTE')
            inputs.extend(_sents)
            targets.extend(_targets)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'E2E')
            inputs.extend(_sents)
            targets.extend(_targets)
            _sents, _targets = get_transformed_io(JD_sents, JD_targets, 'AOPE')
            inputs.extend(_sents)
            targets.extend(_targets)
        else:
            sents, _targets = read_line_examples_from_file(self.args.test_dataset)
            inputs, targets = get_transformed_io(sents, _targets, 'ASQP')

        for i in range(len(inputs)):
            # change input and target to two strings
            input  = inputs[i]
            target = targets[i]
            if len(input) > max_input_len:
                max_input_len = len(input)
            if len(target) > max_target_len:
                max_target_len = len(target)

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_input_max,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
        print(f'max_input_len:{max_input_len}; max_target_len:{max_target_len}')