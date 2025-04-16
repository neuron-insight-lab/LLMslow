import argparse
import os
import json
import logging
import gc

import torch
from tqdm import tqdm
import time
from transformers import set_seed

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='llama2',
                        choices=MODEL_PATHS.keys())
parser.add_argument("--data_name", type=str, default="alpaca",
                    choices=['sharegpt', 'alpaca', 'squad', 'hellaswag', 'mbpp', 'truthful_qa', 'gsm8k', 'emotion', 'OpenHermes'])
parser.add_argument('--max_length', default=1024, type=int)
parser.add_argument("--seed", type=int, default=23, help='random seed')


args = parser.parse_args()


set_seed(args.seed)
save_dir = 'res/Normal/' + str(args.model_name) + '_' + str(args.data_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
args.save_dir = save_dir

model_path = MODEL_PATHS[args.model_name]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# low_cpu_mem_usage=True, use_cache=False,
model, tokenizer = load_model_and_tokenizer(model_path, device=device)

model.generation_config.max_length = args.max_length

# load dataset
data = read_data(args.data_name)

start_epoch = 0

# Check if you've trained before
is_before = len(os.listdir(args.save_dir)) > 0
if is_before:
    path = sorted(os.listdir(args.save_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))[-1]
    start_epoch = int(path.split('.')[0].split('_')[-1])
    # print(path, start_epoch)


for epoch in tqdm(range(start_epoch, len(data))):
    res = dict()
    t1 = time.time()
    # print(f'\n========epoch {epoch} / {len(data)}========')
    prompt = data[epoch]
    save_path = os.path.join(args.save_dir, f'res_{epoch}.json')


    input_ids = get_chat_prompt(tokenizer, prompt, add_generation_prompt=True, return_tensors='pt')
    is_success, success_rate, avg_len, avg_ppl, answer = test_suffix(model, tokenizer, input_ids)
    res[epoch] = {
        'adv_prompt': prompt,
        'answer': answer,
        'avg_len': avg_len,
        'avg_ppl': avg_ppl,
        'success_rate': success_rate,
        'time': time.time() - t1
    }
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=4)