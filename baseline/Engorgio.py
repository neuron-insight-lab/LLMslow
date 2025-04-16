import os
import torch
import argparse
import json
import time
import torch.nn.functional as F

from transformers import set_seed

import sys
sys.path.append('.')
from utils import MODEL_PATHS, load_model_and_tokenizer, read_data, test_suffix

'''
    source code from https://github.com/jianshuod/Engorgio-prompt
'''

class TemplateFactory():
    '''
        1. Use a sentence to get template and then encode
        2. Extract the template part of the encoded
    '''
    def __init__(self, model_name, trigger_token_length, tokenizer, embedding, user_prompt='') -> None:
        self.model_name = model_name
        self.trigger_token_length = trigger_token_length
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.add_additional_prompt(user_prompt)
    

    def add_additional_prompt(self, user_prompt):
        
        if user_prompt != '': 
            user_prompt += ' '
        demo_sentence = self.tokenizer.decode([7993] * self.trigger_token_length).strip()
        message = [
                {'role': 'user', 'content': user_prompt + demo_sentence}
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        self.set_template(prompt)


    def set_template(self, prompt):
        tokenizer = self.tokenizer
        trigger_token_length = self.trigger_token_length
        embedding = self.embedding

        input_ids = tokenizer.encode(prompt)
        print(prompt)
        print(input_ids)
        total_length = len(input_ids)

        prefix_len = max(index for index, item in enumerate(input_ids) if item == 7993) - trigger_token_length + 1
        self.prefix_tokens = input_ids[:prefix_len]
        self.tail_tokens = input_ids[prefix_len+trigger_token_length:]
        
        self.prefix_embeds = embedding[input_ids[:prefix_len]].detach().unsqueeze(0)
        self.tail_embeds = embedding[input_ids[prefix_len+trigger_token_length:]].detach().unsqueeze(0)
        
        self.template_length = total_length - trigger_token_length
        self.response_offset = prefix_len+trigger_token_length
        self.prefix_length = prefix_len
        self.template_w_trigger_length = total_length


    def get_input_embeddings(self, inputs_embeds):
        front_part = inputs_embeds[:, :self.trigger_token_length]
        tail_part = inputs_embeds[:, self.trigger_token_length:]
        concated = torch.concat(
            [self.prefix_embeds, front_part, self.tail_embeds, tail_part], dim=1)
        return concated
    

    def get_input_tokens(self, inputs_tokens):
        return self.prefix_tokens + inputs_tokens + self.tail_tokens


class Converge_check():
    def __init__(self) -> None:
        self.counter = 0
        self.new_state = None
    

    def __call__(self, new_trigger_tokens):
        if self.new_state is None: 
            self.new_state = new_trigger_tokens
            return False

        if self._check_the_same(self.new_state, new_trigger_tokens):
            self.counter += 1
        else:
            self.new_state = new_trigger_tokens
            self.counter = 0
        if self.counter >= 50: return True
        else: return False


    def _check_the_same(self, l1, l2):
        res = True
        dis_list = []
        for idx, (l1x, l2x) in enumerate(zip(l1, l2)):
            if l1x != l2x: 
                res = False
                dis_list.append(idx)
        return res


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = MODEL_PATHS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)

    model.generation_config.max_length = args.max_length

    try:
        total_vocab_size = model.get_output_embeddings().out_features
    except:
        total_vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab().keys())

    args.total_vocab_size = total_vocab_size
    args.eos_token_id = model.config.eos_token_id
    print("eos_token_id:", args.eos_token_id)

    # load dataset
    data = read_data(args.data_name)

    for epoch in range(len(data)):
        print(f'\n\n========epoch {epoch} / {len(data)}========')
        prompt = data[epoch]

        attack(model, tokenizer, args, prompt, epoch)


def attack(model, tokenizer, args, prompt, epoch):
    device = model.device
    total_vocab_size = args.total_vocab_size

    save_path = os.path.join(args.save_dir, f'res_{epoch}.json')

    # emb = model.get_input_embeddings()
    embeddings = model.get_input_embeddings()(torch.arange(0, total_vocab_size).long().to(device)).detach()
    # prefix_sentence = "StableLM is a helpful and harmless open-source AI language model developed"
    trigger_seq_length = args.trigger_token_length

    # -----------------[Init the Env]------------------ 
    # prompt = "Where is the capital of France?"
    template_fac = TemplateFactory(
        args.model_name, trigger_seq_length, tokenizer, embeddings, user_prompt=prompt
    )

    template_len = template_fac.template_length
    theta_length = args.max_length - template_len
    args.theta_length = theta_length
    checker = Converge_check()

    # -----------------[Init the Trigger Theta]------------------ 
    log_coeffs = torch.zeros(theta_length, total_vocab_size, 
                             dtype=embeddings.dtype, device=device, requires_grad=True)

    # -----------------[Training]------------------ 
    optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
    loss_options = args.loss_opt
    args.alpha = args.opt_alpha
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr
    s_time = time.time()
    global_triggers = []
    # losse1s = []; losse2s = []
    for i in range(args.num_iters):        
        optimizer.zero_grad()

        coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0), hard=False)

        # if args.load_in_8bit: coeffs = coeffs.half()
        inputs_embeds = (coeffs @ embeddings)
        inputs_embeds_x = template_fac.get_input_embeddings(inputs_embeds).to(device)

        pred = model(inputs_embeds=inputs_embeds_x).logits
        pred_t = pred.contiguous().view(-1, total_vocab_size)
        target_t = F.softmax(log_coeffs, dim=1)

        loss, (loss1, loss2) = total_loss_v1(pred_t, target_t, template_fac, args, need_detail=True, loss_options=loss_options)
        # losse1s.append([(i + 1), loss1]); losse2s.append([(i + 1), loss2])
        loss.backward()
        optimizer.step()

        trigger_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()[:trigger_seq_length]
        if checker(trigger_ids): break
        print(f'[Epoch {i}]({checker.counter}), loss:{loss.item()}, CEL:{loss1}, WOL:{loss2}')
        if (i + 1) % args.log_interval == 0: 
            global_triggers.append(template_fac.get_input_tokens(checker.new_state))
            print(f"Time Cost: {time.time() - s_time}", tokenizer.decode(trigger_ids, skip_special_tokens=True))

    if (i + 1) % args.log_interval != 0: 
        global_triggers.append(template_fac.get_input_tokens(checker.new_state))
    # if args.save_path != '': torch.save(global_triggers, os.path.join(config.save_dir, f"{args.save_path}.pth"))

    # -----------------[Evaluation]------------------ 
    torch.cuda.empty_cache()
    args.control_slice = slice(template_fac.prefix_length, template_fac.response_offset)

    res = eval_triggers(global_triggers, args, device, (tokenizer, model)) 

    res[-1] = {'initial_prompt': prompt}
    # -----------------[save]------------------   
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=4)


def total_loss_v1(pred_t, target_t, template_fac:TemplateFactory, args, need_detail=False, loss_options=[1, 2]):
    '''
        pred_t     (max_length, V)
        target_t   (max_length - template_length, V)
    '''
    prefix_length = template_fac.prefix_length
    trigger_seq_length = template_fac.trigger_token_length
    response_length = template_fac.response_offset
    output_length = template_fac.template_w_trigger_length
    theta_length = args.max_length - template_fac.template_length

    # Self-Mentor Loss
    # Part 1 -- Trigger Part [prefix_length + 1, response_offset)
    loss1_1 = SelfMentorLoss(pred_t[prefix_length:response_length - 1], target_t[1:trigger_seq_length])
    # Part 2 -- Output Part
    loss1_2 = SelfMentorLoss(pred_t[output_length - 1:-1], target_t[trigger_seq_length:])
    loss1 = (loss1_1 + loss1_2) / theta_length

    if args.esc_loss_version == 0:
        loss2 = EOSProbLoss(pred_t[prefix_length:], args) # v0
    else:
        loss2 = EOSProbLoss(pred_t[output_length:], args) # v1
    loss = 0
    if 1 in loss_options: loss += loss1
    if 2 in loss_options: loss += loss2

    return  loss, (loss1.item(), loss2.item()/args.alpha) if need_detail else ()


def SelfMentorLoss(pred, target, reduction='sum'):
    log_likelihood = -F.log_softmax(pred, dim=1)
    batch = pred.shape[0]
    cel_list = torch.mul(log_likelihood, target)
    if reduction == 'average':
        loss = torch.sum(cel_list) / batch
    else:
        loss = torch.sum(cel_list)
    return loss


def EOSProbLoss(pred_t, args):
    normalized_pred_t = F.softmax(pred_t, dim=1)
    loss = torch.sum(normalized_pred_t.view(-1, args.total_vocab_size)[:, args.eos_token_id])
    return args.alpha * loss


def _eval_triggers(trigger_list, args, device, model_existing=None):
    batch_size = args.batch_size
    sample_time = args.sample_time
    max_length = args.max_length

    control_slice = args.control_slice
    if model_existing is not None:
        tokenizer, model = model_existing

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    res = dict()
    for i, trigger_tokens in enumerate(trigger_list):
        adv_prompt = tokenizer.decode(trigger_tokens[control_slice], skip_special_tokens=True)
        print("=="*40)
        print(adv_prompt)
        print("=="*40)

        remaining_samples = sample_time
        cnt, sum_max = 0, 0

        s_time = time.time()
        length_list = []
        while remaining_samples > 0:
            bs = min(remaining_samples, batch_size)
            remaining_samples -= bs
            cnt += 1

            trigger_tokens_tensor = torch.tensor([trigger_tokens]).repeat(bs, 1).to(device)
            attention_mask = torch.ones_like(trigger_tokens_tensor)
            out = model.generate(
                input_ids=trigger_tokens_tensor, 
                attention_mask=attention_mask,
                max_length=max_length, 
                pad_token_id=pad_token_id,
            )

            for x in out:
                cnt_len = torch.sum((x != pad_token_id).int()).item()
                
                length_list.append(cnt_len)
                if max_length - cnt_len < 10 :
                    sum_max += 1
                    print(tokenizer.decode(x, skip_special_tokens=True))

                length_list.append(cnt_len)
                print(f'-------------------{cnt_len}---------------------')
            print(f"Part {cnt}, Time Cost: {time.time() - s_time}")
        sum_time = time.time() - s_time
        avg_time = sum_time / sample_time
        avg_len = sum(length_list) / len(length_list)
        success_rate = sum_max / sample_time
        res[i] = {
            'adv_prompt': adv_prompt,
            'avg_time': avg_time,
            'avg_len': avg_len,
            'success_rate': success_rate,
        }
    return res



def eval_triggers(trigger_list, args, device, model_existing=None):
    

    control_slice = args.control_slice
    if model_existing is not None:
        tokenizer, model = model_existing

    res = dict()
    for i, trigger_tokens in enumerate(trigger_list):
        adv_prompt = tokenizer.decode(trigger_tokens[control_slice], skip_special_tokens=True)
        print("=="*40)
        print(adv_prompt)
        print("=="*40)

        trigger_tokens = torch.tensor([trigger_tokens])
        is_success, success_rate, avg_len, avg_ppl, answer = test_suffix(model, tokenizer,
                             trigger_tokens, batch=args.batch_size, sample_times=args.sample_time)
        
        res[i] = {
            'adv_prompt': adv_prompt,
            'answer': answer,
            'avg_len': avg_len,
            'avg_ppl': avg_ppl,
            'success_rate': success_rate,
        }
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # [Basic], Experiment Settings
    parser.add_argument('--model_name', default='qwen2.5-1.5b',
                        choices=MODEL_PATHS.keys())
    parser.add_argument("--data_name", type=str, default="alpaca",
                        choices=['sharegpt', 'alpaca', 'squad', 'hellaswag', 'mbpp', 'truthful_qa', 'gsm8k', 'emotion', 'OpenHermes'])

    parser.add_argument("--seed", default=23, type=int, help="Trial Seed")
    parser.add_argument("--log_interval", default=100, type=int, help="Every x iters, eval the theta")

    # [Training], Design Settings
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--num_iters", default=300, type=int, help="number of epochs to train for")
    parser.add_argument("--opt_alpha", default=1, type=float, help="weight of the wiping out loss")
    parser.add_argument("--loss_opt", type=int, nargs='+', default=[1, 2])
    parser.add_argument("--esc_loss_version", default=0, type=int)

    # [Initialization], Theta Settings
    parser.add_argument("--trigger_token_length", default=20, type=int, help='how many subword pieces in the trigger')
    parser.add_argument("--max_length", default=1024, type=int)

    # [Inference], Evaluation Settings
    parser.add_argument("--batch_size", default=4, type=int, help="[Inference], batch size for inference")
    parser.add_argument("--sample_time", default=16, type=int, help="[Inference], total sample time to calculate avg_rate")

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    save_dir = 'res/Engorgio/' + str(args.model_name) + '_' + str(args.data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.save_dir = save_dir
    main(args)
