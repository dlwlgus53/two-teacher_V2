
import copy
import pdb
import sys
import torch
import random
import numpy as np
import json
import progressbar
import ontology
import random
import argparse
from transformers import AutoTokenizer
import config as cfg
from utils import make_label_key, dictionary_split
from collections import defaultdict
import re
from logger_conf import CreateLogger
import os

class VerifyData:
    def __init__(self, tokenizer, data_path, data_type, log_folder, neg_nums=0, short=0, upsamp=False, ):
        self.logger = CreateLogger(f'data_{data_type}', os.path.join(log_folder, "info.log"))
        
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = json.load(open(data_path, "r"))
        self.data_type = data_type
        self.short = short

        self.neg_nums = neg_nums
        self.upsamp = upsamp


        dial_id, turn_id, question, answer, belief, pseudo, ans_type = self.seperate_data(
            self.raw_dataset)
        assert len(question) == len(ans_type)
        
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.belief = belief
        self.pseudo = pseudo
        self.ans_type = ans_type



    def __len__(self):
        return len(self.question)

    def remove_unuse_domain(self, dst):
        new_dst = {}
        for key in dst:
            if key in ontology.all_domain:
                new_dst[key] = dst[key]
        return new_dst

    def make_bspn(self, dict_bspn):
        ans = []  # un usual
        for domain_slot in ontology.all_domain:
            if domain_slot in dict_bspn:
                domain, slot = domain_slot.split(
                    "-")[0], domain_slot.split("-")[1]
                if ("[" + domain + ']') not in ans:
                    ans.append("[" + domain + ']')
                ans.append(slot)
                ans.append(dict_bspn[domain_slot])
        ans = ' '.join(ans)
        return ans
    
    def make_dial(self, org_dial, new_turn_text):
        new_dial = org_dial[:org_dial.rfind(cfg.USER_tk)]
        new_dial += cfg.USER_tk
        new_dial += new_turn_text
        return new_dial
    
    def make_value_dict(self, dataset):
        values = defaultdict(list)
        for dial in dataset:
            for t_id, turn in enumerate(dial):
                if 'belief' not in turn:
                    self.logger.info(f"no belief {turn['dial_id']} {t_id}")
                    continue
                for key_idx, key in enumerate(ontology.all_domain): 

                    if key in turn['belief']:
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list):
                            # in muptiple type, a == ['sunday',6]
                            belief_answer = belief_answer[0]
                        values[key].append(belief_answer)
                        values[key] = list(set(values[key]))
        try:
            values['restaurant-time'].remove('11:30 | 12:30')
        except ValueError:
            pass
        try:
            values['hotel-pricerange'].remove('cheap|moderate')
        except ValueError:
            pass
        return values
    
    def make_dial_dict(self, dataset):
        dial_dict =  defaultdict(list)
        for dial in dataset:
            for turn in dial:
                turn_domain = list(set([k.split("-")[0] for k in turn['curr_belief'].keys()]))
                for dom in turn_domain:
                    dial_dict[dom].append((turn['user'].replace("<sos_u>","").replace("<eos_u>","").strip(), turn['belief']))
        return dict(dial_dict) 
    
    
    def aug_dial(self, dst, dial_dict, seed, neg_nums):
        random.seed(seed)
        def dial_replace(dst, dial_dict):
            if len(dst)>0:
                domain = random.choice(list(dst.keys())).split("-")[0]
            else:
                domain = 'hotel'
            random_dial, random_dst = random.choice(dial_dict[domain])
            cnt = 0
            while dst == random_dst and cnt<3:
                random_dial, random_dst = random.choice(dial_dict[domain])
                cnt +=1
            return random_dial
        
        result = []
        for _ in range(neg_nums):
            result.append(dial_replace(dst,dial_dict))
        return result

    def aug_dst(self, dst, value_dict, seed, neg_nums):
        random.seed(seed)
        def add(dst, value_dict):
            try:
                domain = random.choice(list(dst.keys())).split("-")[0]
                slot = random.choice([item for item in value_dict.keys(
                ) if item.startswith(domain) and item not in dst.keys()])
                value = random.choice(value_dict[slot])
                dst[slot] = value
            except IndexError as e:
                dst = dst

            return dst

        def delete(dst):
            try:
                slot = random.choice(list(dst.keys()))
                del dst[slot]
            except IndexError as e:
                dst = None
            return dst

        def replace(dst, value_dict):
            try:
                slot = random.choice(list(dst.keys()))
                choices = value_dict[slot].copy()
                if dst[slot] in choices:
                    choices.remove(dst[slot])
                dst[slot] = random.choice(choices)
                
            except IndexError as e:
                dst = None
            except:
                pdb.set_trace()
            return dst
        
        result = []
        for _ in range(neg_nums):
            temp = [(add(dst.copy(), value_dict),"aug_a"), (delete(dst.copy()),"aug_d"), (replace(dst.copy(), value_dict),"aug_r")] 
            result.extend(temp)

        result = [i for i in result if i[0] is not None]
        return result

    def seperate_data(self, dataset):
        # TODO See all the history, Not just current history
        value_dict = self.make_value_dict(dataset)
        dial_dict = self.make_dial_dict(dataset) # [domain :[(dial, belief)]]
        
        question, pseudo = [], []
        answer = []
        dial_id = []
        turn_id = []
        belief = []
        ans_type = []
        dial_num = 0
        S = 0
        for dial in dataset:
            S += 1
            if self.short == True and S > 30:
                break
            turn_text = ""
            dial_num += 1
            d_id = dial[0]['dial_id']
            if 'belief' not in dial[0]:
                self.logger.info(f"no belief {d_id}")
                raise ValueError
            for t_id, turn in enumerate(dial):
                turn_text += cfg.USER_tk
                turn_text += turn['user'].replace("<sos_u>","").replace("<eos_u>","").strip()
                turn['belief'] = self.remove_unuse_domain(turn['belief'])
                belief_answer = self.make_bspn(turn['belief'])

                q1 = f"verify the question and answer : context : {turn_text}, Answer : {belief_answer}"
                a1 = 'true'
                b1 = turn['belief']
                p = turn['pseudo']
                
                # 일단 기본으로 정답인거는 여기 둬야하는데
                question.append(q1)
                answer.append(a1)
                dial_id.append(d_id)
                turn_id.append(t_id)
                belief.append(b1)
                pseudo.append(p)
                ans_type.append("g") 

                # neg smaple필요한건 training이니까 label type아니면 넘어감. 근데 label type일 땐 neg sample이 필요해
                ###### neg sampling about DST label
                for neg_belief, ans_type_ in self.aug_dst(turn['belief'], value_dict, t_id, self.neg_nums): # here neg sampling
                    wrong_belief_answer = self.make_bspn(neg_belief)
                    q2 = f"verify the question and answer : context : {turn_text}, Answer : {wrong_belief_answer}"
                    a2 = 'false'
                    b2 = neg_belief

                    question.append(q2)
                    answer.append(a2)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                    belief.append(b2)
                    pseudo.append(p)
                    ans_type.append(ans_type_)
                    
                    # upsampling이 필요하다면..
                    if self.upsamp:
                        question.append(q1)
                        answer.append(a1)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        belief.append(b1)
                        pseudo.append(p)
                        ans_type.append("g")


                for neg_turn in self.aug_dial(turn['belief'], dial_dict, t_id, self.neg_nums): # here neg sampling
                    wrong_turn_text = self.make_dial(turn_text, neg_turn)
                    q3 = f"verify the question and answer : context : {wrong_turn_text}, Answer : {belief_answer}"
                    a3 = 'false'
                    b3 = turn['belief']

                    question.append(q3)
                    answer.append(a3)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                    belief.append(b3)
                    pseudo.append(p)
                    ans_type.append("aug_dial")
                    
                    # upsampling이 필요하다면..
                    if self.upsamp:
                        question.append(q1)
                        answer.append(a1)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        belief.append(b1)
                        pseudo.append(p)
                        ans_type.append("g")
                        
                turn_text += cfg.SYSTEM_tk
                turn_text += turn['resp'].replace("<sos_r>","").replace("<eos_r>","").strip()
        
        self.logger.info(f"total dial num is {dial_num}")
        self.logger.info(f"-- org : {ans_type.count('g')}")
        self.logger.info(f"-- aug add : {ans_type.count('aug_a')}")
        self.logger.info(f"-- aug replace : {ans_type.count('aug_r')}")
        self.logger.info(f"-- aug delete : {ans_type.count('aug_d')}")
        self.logger.info(f"-- aug dial : {ans_type.count('aug_dial')}")
        
        return dial_id, turn_id, question, answer, belief, pseudo, ans_type

    def encode(self, texts, return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus(
                    [text], padding=False, return_tensors=return_tensors) 
                if len(tokenized) > self.max_length:
                    idx = [m.start() for m in re.finditer(cfg.USER_tk, text)]
                    text = text[:idx[0]] + text[idx[1]:]  # delete one turn
                else:
                    break

            examples.append(tokenized)
        return examples

    def __getitem__(self, index):
        target = self.target[index]
        question = self.question[index]
        turn_id = self.turn_id[index]
        dial_id = self.dial_id[index]
        belief = self.belief[index]
        pseudo = self.pseudo[index]
        ans_type = self.ans_type[index]
        return {"question": question, "target": target, "turn_id": turn_id, "dial_id": dial_id, 'belief': belief, 'pseudo': pseudo, 'ans_type' : ans_type}

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        input_source = [x["question"] for x in batch]
        target = [x["target"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        belief = [x["belief"] for x in batch]
        pseudo = [x["pseudo"] for x in batch]
        ans_type = [x["ans_type"] for x in batch]

        source = self.encode(input_source)
        source = [{k: v.squeeze() for (k, v) in s.items()} for s in source]
        source = self.tokenizer.pad(source, padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length=self.max_length,
                                                padding=True, return_tensors='pt', truncation=True)

        return {"input": source, "label": target, "turn_id": turn_id, "dial_id": dial_id, 'belief': belief, 'pseudo': pseudo, 'ans_type' : ans_type}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='t5-small')
    parser.add_argument('--labeled_data_path', type=str,
                        default='/home/jihyunlee/two-teacher/pptod/data/multiwoz/data/multi-woz-fine-processed/_0.1_1_b.json')
    parser.add_argument('--test_data_path', type=str,
                        default='/home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json')
    parser.add_argument('--base_trained', type=str,
                        default="t5-small", help=" pretrainned model from 🤗")
    parser.add_argument('--upsamp', type=int, default=0)
    parser.add_argument('--neg_nums', type=int, default=3)
    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = VerifyData(tokenizer, args.labeled_data_path,\
        'train', args.neg_nums, short=1, upsamp=args.upsamp)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn, shuffle = False)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(len(batch['input']['input_ids'])):
            print(t.decode(batch['input']['input_ids'][i],
                    skip_special_tokens=True))
            print(t.decode(batch['label']['input_ids'][i]))
            print(f"Type : {batch['ans_type'][i]}")
            print()
            pdb.set_trace()
