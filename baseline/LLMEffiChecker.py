import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_HOME'] = '/root/autodl-tmp/models/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import argparse
import json
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import datetime

from transformers import set_seed

import sys
sys.path.append('.')
from utils import MODEL_PATHS, load_model_and_tokenizer, read_data, test_suffix


'''
    source code from https://github.com/Cap-Ning/LLMEffiChecker
'''


class WordAttack:
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(self.device)
        # self.generation_config=generation_config
        self.embedding = self.model.get_input_embeddings().weight
        self.specical_token = self.tokenizer.all_special_tokens
        self.specical_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.pad_token_id = self.model.generation_config.pad_token_id
        # self.space_token = space_token

        self.num_beams = config['num_beams']
        self.num_beam_groups = config['num_beam_groups']
        self.max_per = config['max_per']
        self.topk = config['topk']
        self.max_len = config['max_len']
        self.batch_size = config['batch_size']
        # self.source_language = config['src']
        # self.target_language = config['tgt']

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()


    def mutation(self, current_adv_text, grad, input_len):
        new_strings = []
        new_strings_ids = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        base_tensor = current_tensor.clone()
        
        for pos, t in enumerate(current_tensor):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token and tgt_t.ne(t).item():
                        new_base_tensor = base_tensor.clone()
                        new_base_tensor[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_tensor, skip_special_tokens=True)
                        token_ids = self.get_input_ids(candidate_s)
                        if token_ids.size(-1) != input_len:
                            continue

                        new_strings.append(candidate_s)
                        new_strings_ids.append(token_ids)
                        cnt += 1
                        if cnt >= self.topk:
                            break
        return new_strings, torch.cat(new_strings_ids, dim=0)


    def compute_loss(self, output_ids, input_len):

        # one_hot = torch.zeros(
        #     output_ids.shape[1],
        #     self.embedding.shape[0],
        #     device=self.device,
        #     dtype=self.embedding.dtype
        # )
        
        # one_hot.scatter_(1, output_ids, 1)
        # inputs_embeds = (one_hot @ self.embedding).unsqueeze(0)

        # logits = self.model(inputs_embeds=inputs_embeds).logits
        logits = self.model(output_ids).logits

        target_ids = output_ids[0, input_len:]
        logits = logits[0, input_len-1: -1, :]

        if self.pad_token_id != self.eos_token_id:
            logits[:, self.pad_token_id] = 1e-12
        softmax_v = self.softmax(logits)
        eos_p = softmax_v[:, self.eos_token_id]
        if isinstance(self.eos_token_id, list):
            eos_p = eos_p.sum(dim=-1)
        target_p = torch.stack([softmax_v[iii, s] for iii, s in enumerate(target_ids)])
        pred = eos_p + target_p
        pred[-1] = pred[-1] / 2
        loss = self.bce_loss(pred, torch.zeros_like(pred))

        return loss
            

    @torch.no_grad()
    def select_best(self, new_strings, new_strings_ids):
        batch_size = self.batch_size
        seqs = []
        output_ids_list = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            
            input_token = new_strings_ids[st:ed].to(self.device)
            attention_mask = torch.ones_like(input_token)
            output_ids = self.model.generate(input_token, attention_mask=attention_mask, max_length=self.max_len)

            seqs.extend(output_ids.ne(self.pad_token_id).int().sum(dim=-1).tolist())
            # for out in output_ids:
            #     cnt_len = torch.sum((out != self.pad_token_id).int()).item()
            #     seqs.append(cnt_len)

            output_ids_list.extend(output_ids.tolist())
            # seqs.extend(output_ids.tolist())    
        pred_len = np.array(seqs)
        best_idx = pred_len.argmax()
        # assert len(new_strings) == len(pred_len)
        return new_strings[best_idx], output_ids_list[best_idx], new_strings_ids[best_idx:best_idx+1]
    

    def get_input_ids(self, content):
        message = [
            {'role': 'user', 'content': content}
        ]
        input_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')
        return input_ids


    def run_attack(self, text, save_path):
        # assert len(text) == 1
        input_ids = self.get_input_ids(text)
        input_ids = input_ids.to(self.device)
        attn_masks = torch.ones_like(input_ids)
        input_len = input_ids.size(-1)

        output_ids = self.model.generate(input_ids, attention_mask=attn_masks)
        
        init_answer = self.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

        res = dict()
        res[-1] = {
            'prompt': text,
            'answer': init_answer,
            'total_len': output_ids.size(-1)
        }
        # adv_his = [(deepcopy(current_adv_text), deepcopy(current_len), 0.0)]
        # modify_pos = []

        
        for idx in tqdm(range(self.max_per)):
            t1 = time.time()

            loss = self.compute_loss(output_ids, input_len)

            self.model.zero_grad()
            loss.backward()
            grad = self.embedding.grad

            new_strings, new_strings_ids = self.mutation(text, grad, input_len)

            text, output_list, input_ids = self.select_best(new_strings, new_strings_ids)

            is_success, success_rate, avg_len, avg_ppl, answer, bert_score = test_suffix(self.model, self.tokenizer, input_ids, batch=self.batch_size, init_answer=init_answer)
        
            res[idx] = {
                'adv_prompt': text,
                'answer': answer,
                'avg_len': avg_len,
                'avg_ppl': avg_ppl,
                'bert_score': bert_score,
                'success_rate': success_rate,
                'loss': loss.item(),
                'time': time.time() - t1
            }
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=4)
            if is_success: break
            output_ids = torch.tensor(output_list, dtype=output_ids.dtype, device=output_ids.device).unsqueeze(0)


def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = MODEL_PATHS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    print('load model %s successful' % args.model_name)

    # device = model.device
    model.generation_config.max_length = args.max_length

    # load dataset
    data = read_data(args.data_name)

    beam = model.config.num_beams if args.beam is None else args.beam
    config = {
        'num_beams': beam,
        'num_beam_groups': model.config.num_beam_groups,
        'max_per': args.max_per,
        'max_len': args.max_length,
        'topk': args.topk,
        'batch_size': args.once_forward_batch
        # 'src': src_lang,
        # 'tgt': tgt_lang
    }
    # attack_class = WordAttack()
    attack = WordAttack(model, tokenizer, device, config)

    t1 = datetime.datetime.now()
    for epoch in range(len(data)):
        print(f'\n\n========epoch {epoch} / {len(data)}========')
        src_text = data[epoch]
        save_path = os.path.join(args.save_dir, f'res_{epoch}.json')
        attack.run_attack(src_text, save_path)     
            
    t2 = datetime.datetime.now()
    print('total time:', t2 - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    # parser.add_argument('--data', default=0, type=int, help='experiment subjects')
    parser.add_argument('--model_name', default='qwen2.5-1.5b',
                        choices=MODEL_PATHS.keys())
    parser.add_argument("--data_name", type=str, default="alpaca",
                        choices=['sharegpt', 'alpaca', 'squad', 'hellaswag', 'mbpp', 'truthful_qa', 'gsm8k', 'emotion', 'OpenHermes'])
    # parser.add_argument('--attack', default=2, type=int, help='attack type')  # WordAttack
    parser.add_argument('--max_length', default=1024, type=int, help='max length')
    parser.add_argument('--max_per', default=5, type=int, help='maximum number of iterations')
    parser.add_argument('--topk', default=2, type=int, help='The first k candidates are selected each time')
    parser.add_argument('--seed', default=23, type=int, help='random seed')
    parser.add_argument('--beam', default=None, type=int, help='beam size')
    parser.add_argument('--once_forward_batch', default=16, type=int) # decrease this number if you run into OOM.
    parser.add_argument('--eval_times', default=16, type=int) 

    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = 'res/LLMEffi/' + str(args.model_name) + '_' + str(args.data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    # print('cuda device-id: ', os.getenv('CUDA_VISIBLE_DEVICES'))

    main(args)