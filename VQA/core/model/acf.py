# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask,group_prob):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask,group_prob)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask,group_prob=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        if group_prob is not None:
            att_map = att_map * group_prob.unsqueeze(1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.group_attn_img = GroupAttention_two_dim_FC(__C.HIDDEN_SIZE,dropout=0.8)
        self.group_attn_seq =  GroupAttention_one_dim(__C.HIDDEN_SIZE,dropout=0.8)
    def forward(self, x, x_mask,prior):
        #group_prob, G_ij, G_xy = self.group_attn(x, x_mask,prior_ij,prior_xy)
        group_prob, neigh_attn = self.group_attn_seq(x,x_mask,prior)
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask, group_prob)))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))
        return x, group_prob,neigh_attn

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        
        self.group_attn_img = GroupAttention_two_dim_FC(__C.HIDDEN_SIZE,dropout=0.8)
        self.group_attn_seq =  GroupAttention_one_dim(__C.HIDDEN_SIZE,dropout=0.8)
        
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask,prior_ij,prior_xy):
        #group_prob, neigh_attn = self.group_attn(x,x_mask, x_mask,prior)
        group_prob, G_ij, G_xy = self.group_attn_img(x, x_mask,prior_ij,prior_xy)
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask,group_prob)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask,group_prob=None)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        return x, group_prob,G_ij,G_xy

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class GED(nn.Module):
    def __init__(self, __C):
        super(GED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):#x:lang, y:img
        # Get hidden vector
        for enc in self.enc_list:
            
            prior = 0.0
            x, group_attn,prior = enc(x, x_mask, prior)

        for dec in self.dec_list:
            G_ij = 0
            G_xy = 0
            y, Mijxy,G_ij,G_xy = dec(y,x, y_mask, x_mask,G_ij,G_xy)
            # y = dec(y, x, y_mask, x_mask)
        return x, y

    
##Group_ATT    
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
    def __init__(self, d_model,dropout=0.8):
        super(GroupAttention_one_dim, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model*2, 1)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


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
    def __init__(self, d_model,dropout=0.8):
        super(GroupAttention_two_dim_FC, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model*2, 1)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


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
        #Mijxy_L = Mijxy
        Mijxy_L=F.interpolate(Mijxy ,scale_factor=2,mode="nearest")
        Mijxy_L = Mijxy_L.repeat(1,2,1,1,1)
        Mijxy_L = Mijxy_L.reshape(-1, int(seq_len2), int(seq_len2))
       

        #end_time = time()
        #print("forward time:",end_time - begin_time)
        return Mijxy_L,Mij,Mxy