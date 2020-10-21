#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/29
@author yrh

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Embedding', 'LSTMEncoder', 'MLAttention', 'AttentionWeights', 'FastMLAttention', 'MLLinear', 'LabelAttention']


class Embedding(nn.Module):
    """

    """
    def __init__(self, vocab_size=None, emb_size=None, emb_init=None, emb_trainable=True, padding_idx=0, dropout=0.2):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        return emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]


class LSTMEncoder(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, layers_num, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers_num, batch_first=True, bidirectional=True)
        self.init_state = nn.Parameter(torch.zeros(2*2*layers_num, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, **kwargs):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, inputs.size(0), 1])
        cell_init, hidden_init = init_state[:init_state.size(0)//2], init_state[init_state.size(0)//2:]
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx], lengths[idx], batch_first=True)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True)
        return self.dropout(outputs[torch.argsort(idx)])


class MLAttention(nn.Module):
    """

    """
    def __init__(self, labels_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(2*hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = self.attention(inputs).transpose(1, 2).masked_fill(~masks, -np.inf)  # N, labels_num, seq
        attention = F.softmax(attention, -1)
        return attention @ inputs   # N, labels_num, 2 * hidden_size

class LabelAttention(nn.Module):
    """
    
    """
    def __init__(self, labels_num, hidden_size, label_emb=None):
        super(LabelAttention, self).__init__()
        
        label_embedding=torch.FloatTensor(labels_num,768)
        
        if label_emb is None:
            nn.init.xavier_normal_(label_embedding)
        else:
            label_embedding.copy_(label_emb)
        
        self.label_embedding=nn.Parameter(label_embedding)
        
        self.key_attn = nn.Linear(2*hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.key_attn.weight)
        self.query_attn = nn.Linear(768, hidden_size)
        nn.init.xavier_uniform_(self.query_attn.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)
        attn_key = self.key_attn(inputs).transpose(1,2) # N, hidden_size, seq
        
        label_emb = self.label_embedding.expand((attn_key.size(0), self.label_embedding.size(0), self.label_embedding.size(1)))
        label_emb = self.query_attn(label_emb) # N, label_num, hidden_size
        
        similarity = torch.bmm(label_emb, attn_key) # N, label_num, seq
        similarity = similarity.masked_fill(~masks, -np.inf)
        similarity = F.softmax(similarity, dim=-1) # N, label_num, seq
        
        return torch.bmm(similarity, inputs) # N, label_num, 2*hidden_size

class AttentionWeights(nn.Module):
    """

    """
    def __init__(self, labels_num, hidden_size, device_ids=None):
        super(AttentionWeights, self).__init__()
        if device_ids is None:
            device_ids = list(range(1, torch.cuda.device_count()))
        assert labels_num >= len(device_ids)
        group_size, plus_num = labels_num // len(device_ids), labels_num % len(device_ids)
        self.group = [group_size + 1] * plus_num + [group_size] * (len(device_ids) - plus_num)
        assert sum(self.group) == labels_num
        self.emb = nn.ModuleList(nn.Embedding(size, hidden_size, sparse=True).cuda(device_ids[i])
                                 for i, size in enumerate(self.group))
        std = (6.0 / (labels_num + hidden_size)) ** 0.5
        with torch.no_grad():
            for emb in self.emb:
                emb.weight.data.uniform_(-std, std)
        self.group_offset, self.hidden_size = np.cumsum([0] + self.group), hidden_size

    def forward(self, inputs: torch.Tensor):
        outputs = torch.zeros(*inputs.size(), self.hidden_size, device=inputs.device)
        for left, right, emb in zip(self.group_offset[:-1], self.group_offset[1:], self.emb):
            index = (left <= inputs) & (inputs < right)
            group_inputs = (inputs[index] - left).to(emb.weight.device)
            outputs[index] = emb(group_inputs).to(inputs.device)
        return outputs


class FastMLAttention(nn.Module):
    """

    """
    def __init__(self, labels_num, hidden_size, parallel_attn=False):
        super(FastMLAttention, self).__init__()
        if parallel_attn:
            self.attention = nn.Embedding(labels_num + 1, hidden_size, sparse=True)
            nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks, candidates, attn_weights: nn.Module):
        masks = torch.unsqueeze(masks, 1)   # N, 1, L
        attn_inputs = inputs.transpose(1, 2)    # N, hidden, L
        attn_weights = self.attention(candidates) if hasattr(self, 'attention') else attn_weights(candidates)
        attention = (attn_weights @ attn_inputs).masked_fill(~masks, -np.inf)  # N, sampled_size, L
        attention = F.softmax(attention, -1)    # N, sampled_size, L
        return attention @ inputs   # N, sampled_size, hidden_size


class MLLinear(nn.Module):
    """

    """
    def __init__(self, linear_size, output_size):
        super(MLLinear, self).__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.linear_weight1 = nn.Linear(linear_size[0],1)
        nn.init.xavier_uniform_(self.linear_weight1.weight)
        self.linear_weight2 = nn.Linear(linear_size[0],1)
        nn.init.xavier_uniform_(self.linear_weight2.weight)
        self.linear_final = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        for linear in self.linear_final:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(linear_size[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, self_input, label_input):
        factor1 = torch.sigmoid(self.linear_weight1(self_input))
        factor2 = torch.sigmoid(self.linear_weight2(label_input))
        factor1 = factor1 / (factor1+factor2)
        factor2 = 1 - factor1
        
        out = factor1*self_input + factor2*label_input
        out = self.dropout1(out)
        
        for linear in self.linear_final:
            out = F.relu(linear(out))
        return torch.squeeze(self.output(self.dropout2(out)), -1)
