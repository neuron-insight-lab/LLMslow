import json
import os


# root = '/root/autodl-tmp/lxy/gitcode/LLMslow-cel/res/alpaca/cand=128_topk=64_llama2_noanswer'
# root = '/root/autodl-tmp/lxy/gitcode/LLMslow-cel/res/engorgio/llama2_alpaca'
# root = '/root/autodl-tmp/lxy/gitcode/LLMslow-cel/res/LLMEffi/llama2_alpaca'
# root_dir = '/root/autodl-tmp/lxy/gitcode/LLMslow-cel/res/Normal'
root_dir = '/root/autodl-tmp/lxy/gitcode/myLLMslow/res/LLMslow_82_64_eos'
# root_dir = '/root/autodl-tmp/lxy/gitcode/myLLMslow/res/LLMEffi'

MAX_LENGTH = 1024


def eval_res(res):
    max_len = 0
    ori_len = 0
    min_ppl = 1e9
    bert_score = 0
    for key, item in dict(res).items():
        if int(key) < 0:
            ori_len = item.get('total_len', 0)
            continue
        # if item['avg_rate'] >= 0.125:
        if item['avg_len'] > max_len:
            max_len = item['avg_len']
            min_ppl = item['avg_ppl']
            bert_score = item['bert_score']
        if item['success_rate'] >= 0.125:
            return 1, MAX_LENGTH, ori_len, int(key), min_ppl, bert_score
        
    return 0, max_len, ori_len, int(key), min_ppl, bert_score


for root in os.listdir(root_dir):
    model_name, data_name = root.split('_')
    print(f'========={model_name}========={data_name}=========')

    total, count = 0, 0
    total_len = 0
    ori_len = 0
    root = os.path.join(root_dir, root)
    ppl = 0
    bert_score = 0
    
    for p in os.listdir(root):
        path = os.path.join(root_dir, root, p)
        if os.path.isdir(path):
            print('dir:', p)
            continue
        if int(p.split('.')[0].split('_')[1]) >= 22:
            # print(p)
            continue
        
        if p.endswith('.json'):
            total += 1
            with open(path) as f:
                j = json.load(f)

            _count, _len, _ori_len, step, min_ppl, _bert_score = eval_res(j)
            # _count, _len = eval_LLMslow(j)
            count += _count
            total_len += _len
            ori_len += _ori_len
            ppl += min_ppl
            bert_score += _bert_score
            # print(step)

    print(f'total={total}\tcount={count}\trate:{count/total}\tavg_len:{total_len/total}')
    print(f'ori_len:{ori_len/total}\tppl:{ppl/total}\tbert_score:{bert_score/total}')
