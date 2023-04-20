import os
import pdb
import sys
import json
import torch
import random
import argparse
from collections import OrderedDict
from logger_conf import CreateLogger

import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from trainer import mwozTrainer
from torch.utils.data import DataLoader

from dataclass_ver import VerifyData
from utils import filter_data, merge_data, make_label_key
from evaluate import acc_metric

parser = argparse.ArgumentParser()

# data setting
parser.add_argument('--labeled_data_path', type=str)
parser.add_argument('--valid_data_path', type=str)
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--verify_data_path', type=str)

# training setting
parser.add_argument('--short',  type=int, default=1)
parser.add_argument('--seed',  type=int, default=1)
parser.add_argument('--max_epoch',  type=int, default=1)
parser.add_argument('--gpus', default=2, type=int,
                    help='number of gpus per node')
parser.add_argument('--save_prefix', type=str,
                    help='prefix for all savings', default='')
parser.add_argument('--patient', type=int,
                    help='prefix for all savings', default=3)
parser.add_argument('--only_labeling', type=int, default=0)

# model parameter
parser.add_argument('--base_trained', type=str,
                    default="t5-small", help=" pretrainned model from 🤗")
parser.add_argument('--fine_trained', type=str)
parser.add_argument('--batch_size_per_gpu', type=int, default=8)
parser.add_argument('--test_batch_size_per_gpu', type=int, default=16)
parser.add_argument('--upsamp', type=int, default=1)
parser.add_argument('--neg_nums', type=int, default=1)


def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def add_original_label(TF_dict, train_path):
    labeled_dataset = json.load(open(train_path, "r"))
    for dial in labeled_dataset:
        for turn_id, turn in enumerate(dial):
            dial_id = turn['dial_id']
            dial_turn_key = make_label_key(dial_id, turn_id)
            assert dial_turn_key not in TF_dict
            TF_dict[dial_turn_key] = 'true'
    return TF_dict


def load_trained(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        k = k.replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def find_test_model(test_output_dir):
    import os
    fileData = []
    for fname in os.listdir(test_output_dir):
        if fname.startswith('epoch'):
            fileData.append(fname)
    fileData = sorted(fileData, key=lambda x: x[-9:])
    return fileData[0:args.test_num]


def save_test_result(test_dataset_path, test_result_dict, save_path):
    test_dataset = json.load(open(test_dataset_path, "r"))
    for dial in test_dataset:
        for turn in dial:
            d_id = turn['dial_id']
            t_id = turn['turn_num']
            turn.update(test_result_dict[d_id][t_id])

    with open(save_path, 'w') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(f"./logs/{args.save_prefix}", exist_ok=True)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./out", exist_ok=True)
    os.makedirs(f"model/optimizer/{args.save_prefix}", exist_ok=True)
    os.makedirs(f"model/{args.save_prefix}",  exist_ok=True)

    init_experiment(args.seed)
    log_folder = f'logs/{args.save_prefix}/'
    logger = CreateLogger('main', os.path.join(log_folder, 'info.log'))
    logger.info("------------------------START NEW TRAINING-----------------")
    if args.short == 1:
        args.max_epoch = 1

    logger.info(args)

    teacher = T5ForConditionalGeneration.from_pretrained(
        args.base_trained, return_dict=True)
    if args.fine_trained:
        teacher = load_trained(teacher,  args.fine_trained)
    teacher = nn.DataParallel(teacher).cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

    # Have to train the verify model
    labeled_dataset = VerifyData(tokenizer, args.labeled_data_path,\
                                'train', args.neg_nums, short=args.short, upsamp=args.upsamp)
    valid_dataset = VerifyData(tokenizer,  args.valid_data_path,
                            'valid',  short=args.short)
    test_dataset = VerifyData(tokenizer,  args.test_data_path,
                            'test',  short=args.short)

    if args.verify_data_path:
        verify_dataset = VerifyData(
            tokenizer,  args.verify_data_path, 'label', short=args.short)

    optimizer_setting = {
        'warmup_init': False,
        'lr': 1e-3,
        'eps': (1e-30, 1e-3),
        'clip_threshold': 1.0,
        'decay_rate': -0.8,
        'beta1': None,
        'weight_decay': 0.0,
        'relative_step': False,
        'scale_parameter': False,
    }

    teacher_optimizer = Adafactor(teacher.parameters(), **optimizer_setting)

    trainer_setting = {
        'train_batch_size': args.batch_size_per_gpu * args.gpus,
        'test_batch_size': args.test_batch_size_per_gpu * args.gpus,
        'tokenizer': tokenizer,
        'log_folder': log_folder,
        'save_prefix': args.save_prefix,
        'max_epoch': args.max_epoch,
    }

    teacher_trainer = mwozTrainer(
        model=teacher,
        valid_data=valid_dataset,
        test_data=test_dataset,
        optimizer=teacher_optimizer,
        logger_name='Vteacher',
        evaluate_fnc=acc_metric,
        belief_type=False,
        **trainer_setting)

    if args.only_labeling == 0:  # have to train
        teacher_trainer.work(train_data=labeled_dataset,
                            test=True, save=True, train=True)
        save_test_result(args.test_data_path, teacher_trainer.test_result_dict,
                        f"model/{args.save_prefix}/test.json")

    print("test end")

    if args.verify_data_path:
        raise NotImplementedError
        tf_savepath = args.save_prefix
        tf_dict = teacher_trainer.make_label(data=verify_dataset)
        with open(tf_savepath, 'w') as f:
            json.dump(tf_dict, f, ensure_ascii=False, indent=4)

    print("DONE")
