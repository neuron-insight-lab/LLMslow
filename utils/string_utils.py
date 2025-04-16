import torch
import random
from datasets import load_dataset
import json
import pandas as pd
from bert_score import score


def read_data(dataset_name, length=50):

    if dataset_name == 'alpaca':
        with open('dataset/alpaca_data.json', 'r') as f:
            dataset = json.load(f)
        data = []
        random.shuffle(dataset)
        for ins in dataset:
            prompt = ins['instruction']
            if ins['input'].strip() != '':
                prompt += '\n' + ins['input']
            data.append(prompt)
            if len(data) >= length:
                break
    elif dataset_name == 'sharegpt':
        with open('dataset/share_gpt.json', 'r') as f:
            dataset = json.load(f)
        data = []
        random.shuffle(dataset)
        for ins in dataset:
            prompt = ins[0]['value']
            data.append(prompt)
            if len(data) >= length:
                break
    elif dataset_name == 'squad':
        # 阅读理解任务
        data = load_dataset("rajpurkar/squad", split='train')
        picks = random.sample(range(len(data)), length)
        data = pd.DataFrame(data[picks])
        template = '[CONTEXT]:{0}\n[QUESTION]:{1}\nAnswer the [QUESTION] based on the content in [CONTEXT].'
        data = [template.format(data.loc[i, 'context'], data.loc[i, 'question']) for i in range(length)]

    elif dataset_name == 'truthful_qa':
        # 常识问题
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")
        data = dataset["validation"]['question']
        data = random.sample(data, 100)
    
    elif dataset_name == 'gsm8k':
        # Grade School Math 小学数学问题
        dataset = load_dataset("openai/gsm8k", "main")
        data = dataset["test"]['question'][:100]
        data = random.sample(data, 100)

    elif dataset_name == 'emotion':
        # 常识问题
        dataset = load_dataset("dair-ai/emotion", "split")
        data = dataset["validation"]['text']
        data = random.sample(data, 100)
        template = '{0}\nClasssify the above sentence into neutral, negative, or positive.'
        data = [template.format(text) for text in data]

    # elif dataset_name == 'hellaswag':
    #     # 文本续写任务
    #     dataset = load_dataset("Rowan/hellaswag")
    #     data = dataset["validation"]['ctx'][:length]

    elif dataset_name == 'OpenHermes':
        # dataset = load_dataset("teknium/OpenHermes-2.5")
        # data = dataset["train"]['conversations']
        # count = 0
        # ls = []
        # random.shuffle(data)
        # random.shuffle(data)
        # for d in data: 
        #     # d = data[i]
        #     if d[0]['from'] == 'system':
        #         human = d[1]['value']
        #         llm = d[2]['value']
        #     else:
        #         human = d[0]['value']
        #         llm = d[1]['value']
        #     if len(human.split()) > 30 and len(human.split()) < 100 and len(llm.split()) < 50 and len(llm.split()) > 10:
        #         ls.append(d)
        #         count += 1
        #         if count >= 372:
        #             break
        # print(len(ls))
        # # data = random.sample(data, 100)
        # with open('dataset/OpenHermes.json', 'w') as f:
        #     json.dump(ls, f, indent=4)
        with open('dataset/OpenHermes.json', 'r') as f:
            dataset = json.load(f)
        data = [d[1]['value'] if d[0]['from'] == 'system' else d[0]['value'] for d in random.sample(dataset, 100)]

    elif dataset_name == 'mbpp':
        # 代码生成任务
        with open('dataset/mbpp.json', 'r') as f:
            data = json.load(f)
        data = [text['prompt'].strip() for text in data]
        data = random.sample(data, 100)
        # for text in data:
        #     prompt.append(text['prompt'])
        # print(len(prompt))  # 397
    else:
        raise NotImplementedError
    
    return data[:length]


def get_chat_prompt(tokenizer, user_content, assistant_content=None, add_generation_prompt=False, is_tokenize=True, return_tensors=None):
    message = [
            {'role': 'user', 'content': user_content}
        ]
    if assistant_content is not None:
        message.append({'role': 'assistant', 'content': assistant_content})
        
    prompt = tokenizer.apply_chat_template(message, tokenize=is_tokenize, add_generation_prompt=add_generation_prompt, return_tensors=return_tensors)    #return_tensors="pt"
    return prompt


def generate_str(model, tokenizer, user_prompt):
    # if gen_config is None:
        # model.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # gen_config = model.generation_config

    prompt = get_chat_prompt(tokenizer, user_prompt, add_generation_prompt=True, is_tokenize=False)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    attn_masks = torch.ones_like(input_ids)

    output_ids = model.generate(input_ids, attention_mask=attn_masks)[0]
    
    gen_str = tokenizer.decode(output_ids[input_ids.size(-1): ], skip_special_tokens=True).strip()
    
    return gen_str, len(output_ids[input_ids.size(-1): ]), len(output_ids), output_ids


@torch.no_grad
def test_suffix(model, tokenizer, prompt_ids, init_answer=None, batch=16, sample_times=16):
    assert len(prompt_ids.shape) == 2

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    remain_times = sample_times
    loss = 0
    avg_len = 0
    success_count = 0
    output_ids = None
    output_answer = []

    while remain_times > 0:
        batch_size = min(batch, remain_times)

        input_ids = prompt_ids.repeat(batch_size, 1).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)

        output_answer.extend(tokenizer.batch_decode(out[:, input_ids.size(1):], skip_special_tokens=True))
        
        # 计算ppl
        target_ids = out.clone()
        target_ids[:, :input_ids.size(1)] = -100
        attention_mask = out.ne(pad_token_id).int()

        outputs = model(out, attention_mask=attention_mask, labels=target_ids)
        loss += outputs.loss * batch_size

        # 计算平均长度，以及达到最大值的比例
        x = attention_mask.sum(dim=-1)
        avg_len += x.sum().item()

        success_item = x.ge(model.generation_config.max_length-5)
        if any(success_item):
            output_ids = out[success_item.int().argmax(dim=-1)]
        success_count += success_item.sum().item()

        remain_times -= batch_size

    if output_ids is None:
        output_ids = out[0]
    answer = tokenizer.decode(output_ids[input_ids.size(1):], skip_special_tokens=True)  

    # 计算BERTScore
    reference = [init_answer] * sample_times
    P, R, F1 = score(output_answer, reference, lang="en", verbose=False)
    bert_score = F1.mean().item()

    avg_len = avg_len / sample_times 
    success_rate = success_count / sample_times
    ppl = torch.exp(loss/sample_times).item()

    is_success = success_count >= 2

    return is_success, success_rate, avg_len, ppl, answer, bert_score


def get_nonascii_toks(tokenizer, device='cuda'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    nonascii_toks.extend(tokenizer.all_special_ids)

    # record blank
    token = '* '
    s = set()
    while True:
        t = tokenizer(token, add_special_tokens=False).input_ids
        t = t[-1]
        if t not in s:
            s.add(t)
            # print(tokenizer.decode([t]), t)
            nonascii_toks.append(t)
        else:
            break
        token += ' '
 
    return torch.tensor(nonascii_toks, device=device)


class SuffixManager:
    def __init__(self, model, tokenizer, instruction, args, target=None):

        self.model = model
        self.tokenizer = tokenizer

        self.instruction = instruction
        self.target = target

        self.adv_len = args.adv_len
        self.adv_suffix = ('* ' * self.adv_len).strip()
        self.adv_token_id = self.tokenizer.encode(self.adv_suffix)[-1]
        self.adv_suffix_ids = [self.adv_token_id] * self.adv_len

        # self.specical_token = self.tokenizer.all_special_tokens
        # self.specical_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.pad_token_id = self.model.generation_config.pad_token_id
        
        self.loss_opt = [1,2,3] if args.loss_opt is None else args.loss_opt
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3

        self.init()

    def init(self):
        
        prompt_ids = get_chat_prompt(self.tokenizer, f'{self.instruction} {self.adv_suffix}',
                                      add_generation_prompt=True, return_tensors='pt')[0]

        prefix_len = max(index for index, item in enumerate(prompt_ids.tolist()) if item == self.adv_token_id) - self.adv_len + 1

        self._control_slice = slice(prefix_len, prefix_len + self.adv_len)
        self._target_slice = slice(len(prompt_ids), None)
        self.tail_chat_ids = prompt_ids[prefix_len + self.adv_len:]
        
        assert all([x == self.adv_token_id for x in prompt_ids[self._control_slice]])

        prompt_ids = get_chat_prompt(self.tokenizer, f'{self.instruction} {self.adv_suffix}',
                                      assistant_content=self.target, return_tensors='pt')[0]
        
        self.prefix_ids = prompt_ids[:prefix_len]
        self.tail_all_ids = prompt_ids[prefix_len + self.adv_len:]

        # self._target_slice = slice(len(prompt_ids) + 1, None)   #llama3


    def get_all_ids(self, adv_suffix_ids):

        adv_suffix_ids = torch.tensor(adv_suffix_ids, dtype=self.prefix_ids.dtype)
        
        # input_ids = get_chat_prompt(self.tokenizer, f'{self.instruction} {self.adv_suffix}',
        #                             assistant_content=self.target, return_tensors='pt')[0]
        
        return torch.cat([self.prefix_ids, adv_suffix_ids, self.tail_all_ids])
    
    def get_input_ids(self, adv_suffix_ids):

        adv_suffix_ids = torch.tensor(adv_suffix_ids, dtype=self.prefix_ids.dtype)

        return torch.cat([self.prefix_ids, adv_suffix_ids, self.tail_chat_ids])

    # def update(self, adv_suffix=None, answer=None):
    #     if adv_suffix is not None:
    #         self.adv_suffix = adv_suffix
        
    #     if answer is not None:
    #         self.target = answer
