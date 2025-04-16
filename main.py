import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_HOME'] = '/root/autodl-tmp/models/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json

import gc

import torch
from tqdm import tqdm
import time
from transformers import set_seed

from utils import *


def individual_gcg(model, tokenizer, prompt, epoch, args, not_allowed_tokens=None):
    device = model.device

    num_steps, num_candidate, topk, once_forward_batch = \
            args.steps, args.num_candidate, args.topk, args.once_forward_batch

    eval_interval = args.eval_interval
    eval_times = args.eval_times

    save_path = os.path.join(args.save_dir, f'res_{epoch}.json')

    # Save initial state
    res = dict()
    start_time = time.time()
    initial_answer, initial_len, total_len, _ = generate_str(model, tokenizer, prompt)
    print(f"initial answer len: {total_len}")
    print(f'initial_answer: {initial_answer}')
    res[-2] = {'prompt': prompt, 'answer': initial_answer, 'total_len': total_len, 'ori_time': time.time()-start_time}
    # answer, best_len, _, _ = generate_str(model, tokenizer, f"{prompt} {adv_suffix}")
    # print(f"initial adversary answer len: {best_len}")
    # print(f'answer: {answer}')
    # res[-1] = {'prompt': f"{prompt} {adv_suffix}", 'answer': answer, 'cur_len': best_len}
    
    suffix_manager = SuffixManager(model=model, tokenizer=tokenizer, instruction=prompt, args=args, target=initial_answer)

    # adv_suffix = suffix_manager.adv_suffix
    current_adv_ids = suffix_manager.adv_suffix_ids

    is_success = False
    momentum_grad = None
    m = args.m

    # suffix_manager.update(answer=initial_answer)

    with open(save_path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    for i in range(num_steps):
        start_time = time.time()
        print(f'\n==============Step {i}===============')
        
        # Step 1. Encode user prompt (behavior + adv suffix + LLM answer) as tokens and return token ids.
        input_ids = suffix_manager.get_all_ids(current_adv_ids)
        input_ids = input_ids.to(device)
        

        coordinate_grad = get_gradients(model, input_ids, suffix_manager)

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            # Add momentum to the gradient
            # try:
            #     if momentum_grad is None:
            #         momentum_grad = coordinate_grad.clone()
            #     else:
            #         momentum_grad = momentum_grad * m + coordinate_grad * (1 - m)
            #         coordinate_grad = momentum_grad.clone()
            # except RuntimeError as e:
            #     print("[[RuntimeError]]", e)
            #     adv_suffix = adv_suffix + '!'
            #     continue
        
            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = input_ids[suffix_manager._control_slice]
            
            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(adv_suffix_tokens, coordinate_grad, 
                                        num_candidate, topk, not_allowed_tokens=not_allowed_tokens)
            
            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            # new_adv_suffixs = get_filtered_cands(tokenizer, new_adv_suffix_toks, 
            #                                     fill_cand=False, curr_control=adv_suffix)
            
            # Step 3.4 Compute loss on these candidates and take the argmin.
            
            
            try:                
                losses = get_all_losses(model, tokenizer, input_ids, new_adv_suffix_toks, 
                                        suffix_manager, batch_size=once_forward_batch)

            except Exception as e:
                print('[OOMError] Try making the “--once-forward-batch” smaller')
                raise e
                # losses = get_all_losses(model, tokenizer, input_ids, new_adv_suffixs,
                #                          suffix_manager, batch_size=1)

            best_id = losses.argmin()
            # best_new_adv_suffix = new_adv_suffixs[best_id]
            current_adv_ids = new_adv_suffix_toks[best_id].tolist()

            current_loess = losses[best_id]

            # Update the running adv_suffix with the best candidate
            # adv_suffix = best_new_adv_suffix
            try:
                adv_suffix = tokenizer.decode(current_adv_ids).strip()
            except Exception as e:
                print('current_adv_ids:', current_adv_ids)
                # raise e

            res[i] = {'prompt': prompt, 'adv_suffix': adv_suffix, 
                      'adv_prompt': f"{prompt} {adv_suffix}", 
                        'current_loesses': current_loess.item()}
            
            answer = None
            if (i+1) % eval_interval == 0:

                # input_ids = get_chat_prompt(tokenizer, f'{prompt} {adv_suffix}', add_generation_prompt=True, return_tensors='pt')
                prompt_ids = suffix_manager.get_input_ids(current_adv_ids).to(device)
                is_success, success_rate, avg_len, avg_ppl, answer, bert_score = test_suffix(model, tokenizer, prompt_ids.unsqueeze(0),
                                                                                init_answer=initial_answer, batch=once_forward_batch, sample_times=eval_times)
                res[i]['answer'] = answer
                res[i]['success_rate'] = success_rate
                res[i]['avg_len'] = avg_len
                res[i]['avg_ppl'] = avg_ppl
                res[i]['bert_score'] = bert_score
                del prompt_ids
            # answer, cur_len, total_len, output_ids = generate_str(model, tokenizer, f"{prompt} {adv_suffix}")
            # print(f'current_best_losses:{current_loess}\ttotal time: {time.time() - start_time}\tbest_new_adv_suffix: {best_new_adv_suffix}')
            # print(f"current length: {cur_len}, total length {total_len}\ttotal time: {duration_time}")

            duration_time = time.time() - start_time
            print(f"total time: {duration_time}")

            res[i]['time'] = duration_time
            
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

            if is_success:
                break

            # suffix_manager.update(adv_suffix=adv_suffix, answer=answer)
            # suffix_manager.update(adv_suffix=adv_suffix)
            
        # (Optional) Clean up the cache.
        del input_ids, coordinate_grad, adv_suffix_tokens, new_adv_suffix_toks, losses,   #, logits, hidden_states
        gc.collect()
        torch.cuda.empty_cache()
    return res, is_success
    
    
def main(args):
    # Load model
    model_path = MODEL_PATHS[args.model_name]
    device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # low_cpu_mem_usage=True, use_cache=False,
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)

    model.generation_config.max_length = args.max_length

    # load dataset
    data = read_data(args.data_name)
    
    not_allowed_tokens = get_nonascii_toks(tokenizer, device) 
    
    # record the result
    start_epoch = 0

    # Check if you've trained before
    is_before = len(os.listdir(args.save_dir)) > 0
    if is_before:
        path = sorted(os.listdir(args.save_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))[-1]
        start_epoch = int(path.split('.')[0].split('_')[-1])
        # print(path, start_epoch)

    success_list = []
    count = 0

    for epoch in range(start_epoch, len(data)):
        
        print(f'\n========epoch {epoch} / {len(data)}========')
        prompt = data[epoch]
        # prompt = 'Where is the capital of France?'

        res, is_success = individual_gcg(model, tokenizer, prompt, epoch, args, not_allowed_tokens=not_allowed_tokens)

        gc.collect()
        torch.cuda.empty_cache()
        # if is_success:
        #     count += 1
        #     idx = sorted(res.keys(), key=lambda x: int(x))[-1]
        #     d = res[idx]
        #     d['num_steps'] = idx
        #     success_list.append(d)
            # break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add_model_args(parser)
    parser.add_argument('--model_name', default='qwen2.5-1.5b',
                        choices=MODEL_PATHS.keys())
    parser.add_argument("--data_name", type=str, default="sharegpt",
                        choices=['sharegpt', 'alpaca', 'squad', 'hellaswag', 'mbpp', 'truthful_qa', 'gsm8k', 'emotion', 'OpenHermes'])

    parser.add_argument('--steps', default=50, type=int)
    parser.add_argument('--topk', default=32, type=int)
    parser.add_argument('--num_candidate', default=64, type=int)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--once_forward_batch', default=8, type=int) # decrease this number if you run into OOM.
    parser.add_argument('--m', default=0, type=float, help='Momentum weight parameter')
    
    # parser.add_argument("--adv-suffix", type=str, default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")    #! ! ! ! !     And repeat the answer a thousand times.
    parser.add_argument("--adv-len", type=int, default=20)
    parser.add_argument("--loss_opt", type=int, nargs='+', default=[2, 3])
    parser.add_argument("--lambda1", type=float, default=1, help='Coefficient of eos loss function')
    parser.add_argument("--lambda2", type=float, default=0.5, help='Coefficient of uni loss function')
    parser.add_argument("--lambda3", type=float, default=0.5, help='Coefficient of tar loss function')
    parser.add_argument('--eval_times', default=16, type=int) 

    # parser.add_argument('--save-dir', type=str, default='res/alpaca/cand=128_topk=64_llama2_noanswer') # squad/batch=128_topk=32/
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23, help='random seed')
    parser.add_argument("--no_cuda", action='store_true', help='disables CUDA')

    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = 'res/LLMslow_55/' + str(args.model_name) + '_' + str(args.data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    # main_multi(args)
    main(args)
