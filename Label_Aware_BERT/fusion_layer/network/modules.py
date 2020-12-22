import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['LabelAttention', 'SelfAttention', 'MLinear']

class LabelAttention(nn.Module):
    """
    LA-BERT Label Attention Layer
    """    
    def __init__(self, hidden_size, labels_num, label_emb):
        super(LabelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = labels_num
        self.label_emb = label_emb

        label_embedding = torch.FloatTensor(self.num_labels,self.hidden_size)

        if self.label_emb is None:
            nn.init.xavier_normal_(label_embedding)
        else:
            label_embedding.copy_(self.label_emb)
        
        self.label_embedding = nn.Parameter(label_embedding,requires_grad=False)
        
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.xavier_uniform_(self.key_layer.weight)
        
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.xavier_uniform_(self.query_layer.weight)
        
    def forward(self, inputs, masks):
        
        attn_key = self.key_layer(inputs).transpose(1,2)
        
        label_emb = self.label_embedding.expand((attn_key.size(0),self.label_embedding.size(0),self.label_embedding.size(1)))
        attn_query = self.query_layer(label_emb)
        
        attention = torch.bmm(label_emb, attn_key).masked_fill(~masks, -np.inf)
        attention = F.softmax(attention, -1)
        
        return torch.bmm(attention, inputs)
    
    
class SelfAttention(nn.Module):
    """
    LA-BERT Multi-label Attention Layer
    """    
    def __init__(self, hidden_size, labels_num):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.labels_num = labels_num
        
        self.attention = nn.Linear(self.hidden_size, self.labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)
        
    def forward(self, inputs, masks):
        attention = self.attention(inputs).transpose(1,2).masked_fill(~masks, -np.inf)
        attention = F.softmax(attention, -1)
        
        return torch.bmm(attention, inputs)
    
class MLinear(nn.Module):
    """
    LA-BERT Attention Fusion Layer
    """   
    def __init__(self, hidden_size, label_num):
        super(MLinear, self).__init__()
        self.hidden_size = hidden_size
        self.label_num = label_num
        
        self.linear_weight1 = nn.Linear(self.hidden_size,1)
        nn.init.xavier_uniform_(self.linear_weight1.weight)
        
        self.linear_weight2 = nn.Linear(self.hidden_size,1)
        nn.init.xavier_uniform_(self.linear_weight2.weight)
        
        self.fusion_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        nn.init.xavier_uniform_(self.fusion_linear.weight)
        
        #self.relu = nn.LeakyReLU(0.1)
        self.ln = nn.LayerNorm([self.label_num, self.hidden_size])
        self.dropout = nn.Dropout(0.5)
        
        self.out_linear = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
        
    def forward(self, self_attn, label_attn):
        factor1 = torch.sigmoid(self.linear_weight1(self_attn))
        factor2 = torch.sigmoid(self.linear_weight2(label_attn))
        factor1 = factor1 / (factor1+factor2)
        factor2 = 1 - factor1
        
        out1 = factor1 * self_attn #[batch, label, hidden]
        out2 = factor2 * label_attn #[batch, label, hidden]
        
        out = torch.cat((out1, out2), dim=-1)
        
        out = self.fusion_linear(out)
        out = self.dropout(out)
        out = self.ln(out)
        out = F.gelu(out)
        out = self.out_linear(out)
        
        return torch.squeeze(out, -1)