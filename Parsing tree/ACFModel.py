# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

from time import *
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask, batch_index):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask, batch_index)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, batch_index):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, batch_index)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer,layers_groupattn, N,N_group,if_convert):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.layers_groupattn = clones(layers_groupattn, N_group)
        self.norm = LayerNorm(layer.size)
        self.convert = if_convert
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        G_ij = 0
        G_xy = 0
        if self.convert == 0:
            for layer in self.layers:
                x, Mijxy,G_ij,G_xy = layer(x, mask, G_ij,G_xy)
            for layer in self.layers_groupattn:
                x, Mijxy,G_ij,G_xy = layer(x, mask, G_ij,G_xy)
        if self.convert ==1:
            for layer in self.layers_groupattn:
                x, Mijxy,G_ij,G_xy = layer(x, mask, G_ij,G_xy)
            for layer in self.layers:
                x, Mijxy,G_ij,G_xy = layer(x, mask, G_ij,G_xy)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout, group_attn = None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        if group_attn is not None:
            self.ga_index = 1
            self.group_attn = group_attn
        else:
            self.ga_index = 0

    def forward(self, x, mask, prior_ij = None,prior_xy=None):
        "Follow Figure 1 (left) for connections."
        if self.ga_index == 0:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            group_prob = None
            G_ij = None
            G_xy = None
        else:
            group_prob,G_ij,G_xy = self.group_attn(x, mask.long(),prior_ij,prior_xy)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, group_prob))
        return self.sublayer[1](x, self.feed_forward), group_prob,G_ij,G_xy

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, batch_index):
        prior = 0.0
        for layer in self.layers:
            x, group_attn,prior = layer(x, memory, src_mask, tgt_mask, prior, batch_index)

        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, group_attn, dropout, opt):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.ga_index = getattr(opt, 'ga_index', 1)
        if self.ga_index == 0:
            print('no group attention')
 
    def forward(self, x, memory, src_mask, tgt_mask, prior, batch_index):
        "Follow Figure 1 (right) for connections."
        #print(batch_index.size(0))
        for t in range(batch_index.size(0)):
            if t == 0:
                m = utils.repeat_tensors(batch_index[t], memory[t].unsqueeze(0))
            else:
                m_tmp = utils.repeat_tensors(batch_index[t], memory[t].unsqueeze(0))
                m = torch.cat([m,m_tmp],dim=0)

        if src_mask is not None:
            for t in range(batch_index.size(0)):
                if t == 0:
                    mask = utils.repeat_tensors(batch_index[t], src_mask[t].unsqueeze(0))
                else:
                    mask_tmp = utils.repeat_tensors(batch_index[t], src_mask[t].unsqueeze(0))
                    mask = torch.cat([mask, mask_tmp], dim=0)
            src_mask = mask
        if self.ga_index == 0:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            group_prob = None
        elif self.ga_index != 0:
            group_prob, neigh_attn = self.group_attn(x, tgt_mask, prior)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, group_prob))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), group_prob, neigh_attn

# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) \
#              / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim = -1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn

def attention(query, key, value, mask=None, dropout=None, group_prob=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if group_prob is not None:
        if mask is not None:
            seq_len = query.size()[-2]
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int64), 0)).cuda()
            scores = scores.masked_fill((mask.long() | b) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = p_attn * group_prob.unsqueeze(1)# M element-wise product A
    elif group_prob is None:
        if mask is not None:
            mask = mask[:,:,:,0:scores.size()[3]]
            if mask.size()[0] != scores.size()[0]:
                mask = mask[0].repeat(scores.size()[0],1,1,1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # (M*A)V,M*A

# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=mask,
#                                  dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#              .view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, group_prob=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, group_prob=group_prob)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, opt=None):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu_index = getattr(opt, 'gelu_index', 1)

    def forward(self, x):
        if self.gelu_index:
            return self.w_2(self.dropout(gelu(self.w_1(x))))
        else:
            return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8, opt = None):
        super(GroupAttention, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.hier_add_index = getattr(opt, 'hier_add_index', 1)

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)
        #print(context.shape, eos_mask.shape)
        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int64), 1)).cuda()
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int64), 0)).cuda()
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int64), -1)).cuda()
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).cuda()

        # mask = eos_mask & (a+c) | b
        mask = eos_mask & (a+c)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)  # p_i,i+1
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)  # \hat{a}_i=p_i,i+1* p_i+1,i
        if self.hier_add_index:
            neibor_attn = prior + (1. - prior) * neibor_attn  # a_i^l=\hat{a}_i+a_i^l-1

        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)
        return g_attn, neibor_attn

def context_col(context,col):
    x,y,z = context.size()
    y = int(math.sqrt(y))
    new_context = torch.zeros(x,y,z,requires_grad=False)
    for i in range(y):
        new_context[:,i,:]=context[:,i*y+col,:]
    return new_context.cuda()

def meanpool(input):
    seq_len = int((math.sqrt(input.size()[1]))/2)
    d_model = input.size()[2]
    batchsize = input.size()[0]

    flattened = input.reshape(batchsize,d_model,int(math.sqrt(input.size()[1])),int(math.sqrt(input.size()[1])))
    pooled = nn.AdaptiveAvgPool2d((seq_len, seq_len))(flattened)
    output = pooled.reshape(batchsize,seq_len*seq_len,d_model)
    return output

def mean_matrix(i,j):
    len = j-i
    a = torch.triu(torch.ones(len, len,requires_grad=False))
    b = torch.arange(1, len+1)
    return (a/b).cuda()


def mean_matrix_all(j):
    mean_mask = torch.zeros(j, j, j,requires_grad=False)
    for i in range(0,j):
        len = j - i
        a = torch.triu(torch.ones(len, len,requires_grad=False))
        b = torch.arange(1, len + 1)
        mean_mask[i, i:, i:] = (a / b)

    return mean_mask.cuda()

class GroupAttention_one_dim(nn.Module):
    def __init__(self, d_model,dropout=0.8,opt = None):
        super(GroupAttention_one_dim, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model*2, 1)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.hier_add_index = getattr(opt, 'hier_add_index',1)


    def forward(self, context, eos_mask,prior):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)

        context1 = context.repeat(1,seq_len,1).reshape(-1,seq_len,self.d_model)
        context2 = context1.transpose(1, 2).reshape(-1, self.d_model, seq_len)
        mean_mask_matrix = mean_matrix_all(seq_len).repeat(batch_size,1,1)
        context_mean = torch.bmm(context2,mean_mask_matrix).reshape(-1,self.d_model,seq_len).transpose(1,2)
        context1 = context1.masked_fill(context_mean==0,0)
        input = torch.cat((context1,context_mean),dim=2).reshape(-1,self.d_model*2)
        #nonZeroRows = torch.abs(input).sum(dim=1) > 0
        #input = input[nonZeroRows]
        Mij = torch.sigmoid(self.fc(input)).reshape(batch_size,seq_len,seq_len)
        if self.hier_add_index:
            Mij = prior + (1. - prior) * Mij

        Mij_tri = torch.triu(torch.ones_like(Mij))
        Gij = Mij.masked_fill(Mij_tri == 0, 1)
        Gij = torch.cumprod(Gij, dim=2)

        Gij = torch.triu(Gij)
        GijT = torch.transpose(Gij,1,2)
        Gij = Gij+GijT #Mij = Mji
        Gij_diag = torch.diagonal(Gij,dim1=-1,dim2=-2)-1
        Gij_diag1 = torch.diag_embed(Gij_diag)
        Gij = Gij - Gij_diag1 # Mii =1

        return Gij,Mij

def convert_col(context):
    batch_size, seq_len2, d_model = context.size()
    seq_len = int(math.sqrt(seq_len2))
    context2 = context.reshape(-1, seq_len, seq_len, d_model)
    context3 = context2.permute(0, 2, 1, 3).reshape(batch_size, seq_len2, d_model)
    return context3.cuda()


class GroupAttention_two_dim_FC(nn.Module):
    def __init__(self, d_model,dropout=0.8,opt = None):
        super(GroupAttention_two_dim_FC, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model*2, 1)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.hier_add_index = getattr(opt, 'hier_add_index',1)


    def forward(self, context,mask,prior_ij,prior_xy):

        #begin_time = time()
        batch_size,seq_len2, d_model = context.size()#bs,144,512
        context = self.norm(context)
        seq_len2 = int(seq_len2)

        context = meanpool(context)
        seq_len1 = int(math.sqrt(seq_len2))
        seq_len = seq_len1
        seq_len = int(seq_len1/2)

        context_i = context.reshape(batch_size*seq_len,seq_len,self.d_model)
        context_x = convert_col(context).reshape(batch_size*seq_len,seq_len,self.d_model)

        context1_i = context_i.repeat(1, seq_len, 1).reshape(-1, seq_len, self.d_model)
        context2_i = context1_i.transpose(1, 2).reshape(-1, self.d_model, seq_len)
        mean_mask_matrix = mean_matrix_all(seq_len).repeat(batch_size*seq_len, 1, 1)
        context_mean_i = torch.bmm(context2_i, mean_mask_matrix).reshape(-1, self.d_model, seq_len).transpose(1, 2)
        context1_i = context1_i.masked_fill(context_mean_i == 0, 0)
        input_i = torch.cat((context1_i, context_mean_i), dim=2).reshape(-1, self.d_model * 2)
        Mij = torch.sigmoid(self.fc(input_i)).reshape(batch_size,seq_len, seq_len, seq_len)
        if self.hier_add_index:
            Mij = prior_ij + (1. - prior_ij) * Mij

        Gij = torch.triu(Mij)
        Gij = Mij.masked_fill(Gij == 0, 1)
        Gij = torch.cumprod(Gij, dim=2)
        Gij = torch.triu(Gij)
        GijT = torch.transpose(Gij, 1, 2)
        Gij = Gij + GijT
        Gij_diag = torch.diagonal(Gij, dim1=-1, dim2=-2) - 1
        Gij_diag1 = torch.diag_embed(Gij_diag)
        Gij = Gij - Gij_diag1

        context1_x = context_x.repeat(1, seq_len, 1).reshape(-1, seq_len, self.d_model)
        context2_x = context1_x.transpose(1, 2).reshape(-1, self.d_model, seq_len)
        context_mean_x = torch.bmm(context2_x, mean_mask_matrix).reshape(-1, self.d_model, seq_len).transpose(1, 2)
        context1_x = context1_x.masked_fill(context_mean_x == 0, 0)
        input_x = torch.cat((context1_x, context_mean_x), dim=2).reshape(-1, self.d_model * 2)
        Mxy = torch.sigmoid(self.fc(input_x)).reshape(batch_size,seq_len, seq_len, seq_len)
        Mxy = torch.triu( Mxy )
        if self.hier_add_index:
            Mxy = prior_xy + (1. - prior_xy) * Mxy

        Gxy = Mxy.masked_fill( Mxy  == 0, 1)
        Gxy = torch.cumprod( Gxy , dim=2)
        Gxy = torch.triu(Gxy)
        GxyT = torch.transpose(Gxy, 1, 2)
        Gxy = Gxy + GxyT
        Gxy_diag = torch.diagonal(Gxy, dim1=-1, dim2=-2) - 1
        Gxy_diag1 = torch.diag_embed(Gxy_diag)
        Gxy = Gxy- Gxy_diag1


        Mij_L = Gij.repeat(seq_len, 1, 1, 1, 1, ).permute(1, 3, 4, 2, 0)
        Mxy_L = Gxy.repeat(seq_len, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
        Mijxy = Mij_L*Mxy_L

        Mijxy_L=F.interpolate(Mijxy ,scale_factor=2,mode="nearest")
        Mijxy_L = Mijxy_L.repeat(1,2,1,1,1)
        Mijxy_L = Mijxy_L.reshape(-1, int(seq_len2), int(seq_len2))
        #Mijxy_L= Mijxy_L.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)/
        #.repeat_interleave(2, dim=3).repeat_interleave(2,dim=4)


        #end_time = time()
        #print("forward time:",end_time - begin_time)
        return Mijxy_L,Mij,Mxy

class ACFModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        N_group = self.N_group
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout, opt=self.opt)
        if self.ga_index == 0:
            group_attn = None
        else:
            #group_attn = GroupAttention(d_model, opt= self.opt)
            group_attn = GroupAttention_one_dim(d_model, opt=self.opt)
        if self.vis_ga_index == 0:
            vis_group_attn = None
        else:
            vis_group_attn = GroupAttention_two_dim_FC(d_model, opt=self.opt)
        position = PositionalEncoding(d_model, dropout)

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, None),EncoderLayer(d_model, c(attn), c(ff), dropout, vis_group_attn), N_enc,N_group,0),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), group_attn, dropout, self.opt), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(ACFModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))

        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.N_group = getattr(self.opt, 'N_group', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.all_former = getattr(opt, 'all_former', 0)
        self.word_mask_ix = opt.word_mask_ix
        self.word_end_ix = opt.word_end_ix
        self.ga_index = getattr(opt, 'ga_index', 1)
        self.vis_ga_index = getattr(opt, 'vis_ga_index', 1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1


        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        return att_feats, seq, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, seq_mask=None, batch_index=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks = self._prepare_feature_forward(att_feats, att_masks, seq)
        seq_mask = seq_mask.unsqueeze(1)

        out = self.model(att_feats, seq, att_masks, seq_mask, batch_index)

        outputs = self.model.generator(out)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, batch_index):
        """
        state = [ys.unsqueeze(0)]
        """
        mask_it = fc_feats_ph.new_ones(it.size(0), dtype=torch.long) * self.word_mask_ix
        if len(state) == 0:
            ys = torch.cat([it.unsqueeze(1), mask_it.unsqueeze(1)], dim=1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1), mask_it.unsqueeze(1)], dim=1)
        seq_mask = torch.from_numpy(np.ones(ys.size()).astype('int64')).to(memory.device).unsqueeze(1)
        out = self.model.decode(memory, mask, ys, seq_mask, batch_index)
        return out[:, -1], [ys[:, :-1].unsqueeze(0)]


    def get_logprobs_state(self, it, fc_feats, att_feats, memory, memory_masks, state,
                           output_logsoftmax=1, batch_index=None):
        # 'it' contains a word index
        it = self.embed(it)

        output, state = self.core(it, fc_feats, att_feats,  memory, state, memory_masks, batch_index)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _sample(self, fc_feats, att_feats, att_masks=None, batch_index=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        batch_index[:] = sample_n
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt, batch_index=batch_index)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, memory, memory_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       memory, memory_masks]
                                                                                      )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_ones((batch_size * sample_n, self.seq_length), dtype=torch.long)*self.word_end_ix
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)

        for t in range(0,self.seq_length):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size * sample_n, dtype=torch.long)
            # seq[:, t] = self.word_mask_ix
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, memory, memory_masks, state,
                                                      output_logsoftmax=output_logsoftmax, batch_index=batch_index)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = (it != self.word_end_ix)
            else:
                unfinished = unfinished * (it != self.word_end_ix)
            # it = it * unfinished.type_as(it)
            it = it.masked_fill(unfinished==False, self.word_end_ix)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}, batch_index = None):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        batch_index[:] = sample_n
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_zeros([batch_size], dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                  batch_index=batch_index)

        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
        #                                                                           [p_fc_feats, p_att_feats,
        #                                                                            pp_att_feats, p_att_masks]
        #                                                                           )
        '''  #print fine_grained alignment score
        temp = nn.Parameter(torch.ones([]) * 0.07)
        with torch.no_grad():
            temp.clamp_(0.001, 0.5)

        img_batch = memory.shape[0]  # b
        text_batch = output.shape[0]  # b
        img_len = memory.shape[1]
        text_len = output.shape[1]
        queue_size_finegrain = memory.shape[0]

        text_feat_finegrain = output.view(-1, 512)  # b*15 512
        img_feat_finegrain = memory.view(-1, 512)  # b*144 512
        img_feat_all_finegrain = (img_feat_finegrain)
        text_feat_all_finegrain = (text_feat_finegrain)
        text_feat_all_mask_finegrain = torch.ones(text_batch, text_len).cuda()

        sim_v2t_finegrain = img_feat_finegrain @ text_feat_all_finegrain.t() / temp  # 1296,6
        sim_t2v_finegrain = text_feat_finegrain @ img_feat_all_finegrain.t() / temp  # 6,1296
        sim_v2t_finegrain = sim_v2t_finegrain.view(img_batch, img_len, queue_size_finegrain, text_len).transpose(1,
                                                                                                                 2).contiguous()  # b 128 196 40
        sim_t2v_finegrain = sim_t2v_finegrain.view(text_batch, text_len, queue_size_finegrain, img_len).transpose(1,
                                                                                                                  2).contiguous()  # b 128 40 196
        sim_v2t_finegrain = sim_v2t_finegrain.masked_fill(text_feat_all_mask_finegrain.unsqueeze(0).unsqueeze(2) == 0,
                                                          0)  # b 128 196 40

        sim_v2t_finegrain = sim_v2t_finegrain.max(-1)[0]  # b 128 196
        sim_t2v_finegrain = sim_t2v_finegrain.max(-1)[0]  # b 128 40

        sim_t2v_finegrain = sim_t2v_finegrain.masked_fill(text_feat_all_mask_finegrain.unsqueeze(1) == 0, 0)  # b 128 40
        sim_t2v_mask = text_feat_all_mask_finegrain.sum(-1).view(text_batch, 1)  # b 1
        sim_v2t_finegrain = sim_v2t_finegrain.view(img_batch, queue_size_finegrain, -1).sum(-1) / img_len  # b 128
        sim_t2v_finegrain = sim_t2v_finegrain.view(text_batch, queue_size_finegrain, -1).sum(-1) / sim_t2v_mask  # b 128
        loss_v2t_finegrain = torch.sum(sim_v2t_finegrain, dim=1)
        loss_t2v_finegrain = torch.sum(sim_t2v_finegrain, dim=1)
        ####################################################################################################
        vtc_loss = (loss_v2t_finegrain + loss_t2v_finegrain) / 2
        with open('fine_grain_score_acf.txt', 'a') as f:
            print(str(vtc_loss.item()), file=f)
        '''
        batch_index[:] = beam_size
        self.done_beams = self.beam_search(state, logprobs, batch_index, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs