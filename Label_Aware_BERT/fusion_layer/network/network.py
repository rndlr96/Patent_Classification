#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/12/22
@author rndlr96

"""

import torch.nn as nn

from deepxml.modules import *

from transformers import BertTokenizer, BertPreTrainedModel, AdamW, BertConfig, BertModel

__all__ = ['BertForMultiLabelSequenceClassification']

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, label_emb):
        super(BertForMultiLabelSequenceClassification, self).__init__(config, label_emb=None)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.label_emb = label_emb
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        
        self.self_attn = SelfAttention(self.hidden_size, self.num_labels)
        self.label_attn = LabelAttention(self.hidden_size, self.num_labels, self.label_emb)
        self.linear = MLinear(self.hidden_size, self.num_labels)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        sequence, _ = self.bert(input_ids, attention_mask)
        sequence = self.dropout(sequence) # [batch, sequence, hidden_size]
        
        masks = attention_mask  != 0 # [batch, sequence]
        masks = torch.unsqueeze(masks, 1) # [batch, 1, sequence]
        
        self_attn = self.self_attn(sequence, masks)
        label_attn = self.label_attn(sequence, masks)
        
        return self.linear(self_attn, label_attn)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True    