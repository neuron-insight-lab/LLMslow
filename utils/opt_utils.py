import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# from attack_manager import get_embedding_matrix, get_embeddings


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda', **kwargs):
    trust_remote_code = 'glm' in model_path

    # if 'qwen2.5-1.5b' == model_path:
    #     kwargs['attn_implementation'] = 'flash_attention_2'
        
    
    # torch_dtype=torch.float16, device_map='auto',# trust_remote_code=True,
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **kwargs
    ).eval().to(device)
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id if model.generation_config.pad_token_id is None else model.generation_config.pad_token_id

    if not model.generation_config.do_sample:
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.9
        
    return model, tokenizer


def get_loss(logits, eos_token_id, input_ids, target_slice, loss_opt=[1,2,3], lambda1=1, lambda2=1, lambda3=1):
    assert len(logits.shape) == 3 and len(input_ids.shape) == 2
    loss = 0
    loss_eos, loss_uni, loss_cel = None, None, None
    if 1 in loss_opt: 
        prob = torch.softmax(logits, dim=-1)
        eos_p = prob[:, :, eos_token_id]
        if isinstance(eos_token_id, list):
            eos_p = eos_p.sum(dim=-1)
        loss_eos = nn.BCELoss(reduction='none')(eos_p, torch.zeros_like(eos_p))
        loss_eos = loss_eos.mean(dim=-1)
        loss += lambda1 * loss_eos
    
    if 2 in loss_opt:
        uni_distribution = torch.ones(logits.shape, device=logits.device) / logits.shape[-1]
        loss_uni = F.kl_div(logits.log_softmax(dim=-1), uni_distribution, reduction='none')
        loss_uni = loss_uni.mean(dim=[1, 2])
        loss += lambda2 * loss_uni

    if 3 in loss_opt:
        loss_cel = nn.CrossEntropyLoss(reduction='none')(logits.transpose(1,2), input_ids[:, target_slice])
        loss_cel = loss_cel.mean(dim=-1)
        loss += lambda3 * loss_cel
    
    # sum_eos = torch.sum(eos_p, dim=-1)
    # target_p = torch.stack([logits[iii, s] for iii, s in enumerate(input_ids[target_slice])])
    # target_p = logits.gather(-1, input_ids[:, target_slice].unsqueeze(-1)).squeeze()
    
    # pred = eos_p + target_p
    # pred[:, -1] = pred[:, -1] / 2
    # pred = pred + 1e-7
    # loss_eos = pred.mean(dim=-1)

    return loss, loss_eos, loss_cel, loss_uni


def get_gradients(model, input_ids, suffix_manager):

    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    eos_token_id = suffix_manager.eos_token_id
    loss_opt = suffix_manager.loss_opt
    lambda1 = suffix_manager.lambda1
    lambda2 = suffix_manager.lambda2
    lambda3 = suffix_manager.lambda3

    # embed_weights = get_embedding_matrix(model)
    embed_weights = model.get_input_embeddings().weight
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    
    one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights)
    
    # now stitch it together with the rest of the embeddings
    # embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    embeds = embed_weights[input_ids].detach()
    full_embeds = torch.cat(
        [
            embeds[:control_slice.start, :], 
            input_embeds, 
            embeds[control_slice.stop:, :]
        ], 
        dim=0).unsqueeze(0)
    
    if 'glm' in model.name_or_path:
        outputs = model(input_ids=input_ids.unsqueeze(0), inputs_embeds=full_embeds)
    else:
        outputs = model(inputs_embeds=full_embeds)

    logits = outputs.logits[:, target_slice.start-1 : -1, :]

    loss, loss_eos, loss_cel, loss_uni = get_loss(logits, eos_token_id, input_ids.unsqueeze(0), target_slice,
                                                   loss_opt=loss_opt, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
    print(f'********loss_eos: {loss_eos.item() if loss_eos else 0}********loss_cel: {loss_cel.item() if loss_cel else 0}********loss_uni: {loss_uni.item() if loss_uni else 0}********')
    # loss.squeeze()

    model.zero_grad()
    loss.backward(retain_graph=False)

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    one_hot.grad.zero_()

    # grad = torch.randn_like(one_hot)

    return grad


def model_forward(model, input_ids, suffix_manager, batch_size=64):
    
    target_slice = suffix_manager._target_slice
    eos_token_id = suffix_manager.eos_token_id
    loss_opt = suffix_manager.loss_opt
    lambda1 = suffix_manager.lambda1
    lambda2 = suffix_manager.lambda2
    lambda3 = suffix_manager.lambda3

    losses = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]

        # 根据loss选择  
        outputs = model(input_ids=batch_input_ids)
        # logits = outputs.logits[:, target_slice.start-1: -1, :].to(input_ids.device)
        logits = outputs.logits[:, target_slice.start-1 : -1, :].to(input_ids.device)
        loss, _, _, _ = get_loss(logits, eos_token_id, batch_input_ids, target_slice, 
                                 loss_opt=loss_opt, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        losses.append(loss)
        
        # gc.collect()

    del batch_input_ids, logits
    gc.collect()
    # torch.cuda.empty_cache()
    
    return torch.cat(losses, dim=0)


def get_all_losses(model, tokenizer, input_ids, new_adv_suffix_toks, suffix_manager, batch_size=128):
    
    control_slice = suffix_manager._control_slice

    # if isinstance(new_adv_suffixs[0], str):
    #     max_len = control_slice.stop - control_slice.start
        
    #     test_ids = tokenizer(new_adv_suffixs, add_special_tokens=False, max_length=max_len,
    #                           padding=True, truncation=True, return_tensors="pt").input_ids.to(model.device)
        
    # else:
    #     raise ValueError(f"test_controls must be a list of strings, got {type(new_adv_suffixs)}")
    test_ids = new_adv_suffix_toks.to(model.device)
    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    # attn_mask = (ids != tokenizer.pad_token_id).type(ids.dtype)

    return model_forward(model, ids, suffix_manager, batch_size=batch_size)



def sample_control(control_toks, grad, batch_size, topk, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    
    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),  # TODO 换个不重复的函数
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, fill_cand=False, curr_control=None):
    cands, count = [], 0
    s = set()
    length = len(tokenizer(curr_control, add_special_tokens=False).input_ids)

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        
        if decoded_str != curr_control and \
            len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == length and  \
            decoded_str not in s:
                cands.append(decoded_str)
                s.add(decoded_str)
        else:
            count += 1
    if len(cands) == 0:
        print(curr_control)
        print(control_cand)

    if fill_cand and count > 0 and len(cands) > 0:
        cands = cands + [cands[-1]] * count
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
        
    return cands