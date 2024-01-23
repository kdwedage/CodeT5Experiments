# Adapted from https://github.com/reddy-lab-code-research/StructCoder/blob/master/pretrain_utils.py

import random
import numpy as np
import torch

def add_noise(code_input, tokenizer):
    mask_frac = 0.35
    poisson_lambda = 12

    max_len = code_input.size()[1]
    num_to_mask = int(np.round(max_len * mask_frac))
    mask_lengths = np.random.poisson(poisson_lambda, num_to_mask) 
    mask_lengths[mask_lengths==0] = 1
    
    # Trim to masking budget
    if mask_lengths.sum()>num_to_mask:
        cum_length = np.cumsum(mask_lengths)
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        mask_lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i+1 if (mask_lengths[i]>0) else i # no. of spans to mask
        mask_lengths = mask_lengths[:num_to_mask]
        
    # start indices for span masking, no <CLS> or <SEP>
    indices = np.sort(np.random.permutation(max_len-2)[:num_to_mask]+1) # 1 to max_len-2
    
    # delete, replace with random word, or mask
    del_rep_mask = np.random.randint(0,3,size=num_to_mask)
    
    # replace with MASK, rand tokens
    mask_token_id = tokenizer.mask_token_id
    ids = code_input.reshape(-1)
    ids = ids[torch.isin(ids.cpu(), torch.LongTensor([tokenizer.pad_token_id, tokenizer.cls_token_id, 
                                                tokenizer.sep_token_id, tokenizer.unk_token_id]), invert=True)]
    num_ids = len(ids)
    keep = np.ones(max_len, dtype=bool)
    for i,(start,mask_length,typ) in enumerate(zip(indices,mask_lengths,del_rep_mask)):
        if typ==0: # replace with mask
            code_input[:,start:start+mask_length] = mask_token_id
        elif typ==1: # replace with random token
            end = min(start+mask_length, max_len)
            code_input[:,start:end] = ids[torch.randint(high=num_ids,size=(end-start,))]
        else: # delete
            keep[start:start+mask_length] = False
            
    # can get an error if keep.sum()<3
    if keep.sum()<3:
        keep[-3:] = True
            
    # delete tokens
    code_input = code_input[:, keep]
    
    return code_input