import json
import os
import torch
import transformers as tfs
import random
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import copy
import math
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')
batch_size=16
max_seq_len=512
MODELNAME='hfl/chinese-macbert-base' 
num_class=10


def encode_mask(src,pad=0):
    src_mask=(src==pad).unsqueeze(-2)  #返回一个true/false矩阵，size = [batch , 1 , src_L]
    return src_mask

class BertClassificationModel(nn.Module):
    """Bert分类器模型"""
    def __init__(self, predicted_size, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        self.hidden_size=hidden_size
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(MODELNAME)
        self.bert = model_class.from_pretrained(MODELNAME)
        # self.bilstm=nn.LSTM(768,128,num_layers=2,batch_first=True,bidirectional=True,dropout=0.2)
        self.mha=MultiHeadedAttention(8,hidden_size,0.2)
        self.ff=PositionwiseFeedForward(hidden_size,hidden_size*2,0.2)
        self.transform=EncoderLayer(size=hidden_size,self_attn=self.mha,feed_forward=self.ff,dropout=0.2)
        self.linear1 = nn.Linear(hidden_size, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(256, predicted_size)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences,\
                              add_special_tokens=True,\
                              truncation=True,\
                              max_length=512,\
                              padding=True)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        mask=encode_mask(attention_mask)

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_output = bert_output[0]
        transform_out=self.transform(bert_output,mask=mask).cuda()
#         linear1_out=self.dropout1(self.linear1(transform_out[:,-1,:].squeeze(0)).cuda()).cuda()
        linear1_out=self.dropout1(self.linear1(torch.mean(transform_out,1).cuda())).cuda()
        linear2_out=self.dropout2(self.linear2(linear1_out).cuda()).cuda()
        return linear2_out

    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  #检测word embedding维度是否能被h整除
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # 头的个数
        self.linears = clones(nn.Linear(d_model, d_model), 4) #四个线性变换，前三个为QKV三个变换矩阵，最后一个用于attention后
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) 
        nbatches = query.size(0) 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view中给-1可以推测这个位置的维度
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
    p_attn = F.softmax(scores, dim = -1) 
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.features=features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features)) 
        self.eps = eps 

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)