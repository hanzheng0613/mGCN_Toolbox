import torch.nn as nn
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch import Tensor
from typing import Union
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

class GCN_Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, weight, dropout=0.5, k=5):
        super(GCN_Net, self).__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.dropout = dropout
        self.weight = weight
        self.k = k

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, edges, edges_neg, common_neighbors):
        x, edge_index = data.x, data.edge_index

        if self.weight:
            weight = data.graph['weight']
            x = F.relu(self.conv1(x, edge_index, weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, weight)
        else:
            # x = self.linear(x)
            # x = self.conv2(x, edge_index)
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        results = torch.cat(((x[edges[:, 0]]* x[edges[:, 1]]).sum(dim=-1),
                             (x[edges_neg[:, 0]]* x[edges_neg[:, 1]]).sum(dim=-1)))

        return results


class Attention(nn.Module):
    def __init__(self, input_size, cuda=False):
        super(Attention, self).__init__()

        self.cuda = cuda
        self.bilinear_layer = nn.Bilinear(input_size, input_size, 1)
        self.softmax = nn.Softmax(dim=1)
        if self.cuda:
            self.bilinear_layer.cuda()
            self.softmax.cuda()

    def forward(self, Ws):
        """
        Measuring relations between all the dimensions
        """

        # W = torch.Tensor(output_size,output_size)

        num_dims = len(Ws)

        attention_matrix = torch.empty((num_dims, num_dims), dtype=torch.float)
        if self.cuda:
            attention_matrix = attention_matrix.cuda()
        for i, wi in enumerate(Ws):
            for j, wj in enumerate(Ws):
                # attention_matrix[i,j] = torch.trace(wi.transpose(0,1).mm(W).mm(wj))
                attention_matrix[i, j] = torch.sum(self.bilinear_layer(wi, wj))

        attention_matrix_softmax = self.softmax(attention_matrix)

        return attention_matrix_softmax


class Combine(nn.Module):
    """
    Combine embeddings from different dimensions to generate a general embedding
    """
    def __init__(self, input_len=6, output_size = 3,cuda=False, dropout_rate=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(Combine,self).__init__()
        self.cuda = cuda
        self.input_len = input_len
        self.output_size = output_size
        self.linear_layer = nn.Linear(self.input_len,self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        #print('input_len type: ',type(input_len))
        #print('output_size type : ',type(output_size))
        self.act = nn.ELU()
    def forward(self,dim_embs):

        emb = torch.cat(dim_embs,1)
        emb_combine = self.linear_layer(emb)
        emb_combine_act = self.act(emb_combine)
        #emb_combine_act_drop = self.dropout(emb_combine_act)

        return emb_combine_act

class Combine_Attention(nn.Module):
    """
    Combine embeddings from different dimensions to generate a general embedding
    """
    def __init__(self, input_len=6, output_size = 3,cuda=False, dropout_rate=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(Combine_Attention,self).__init__()
        # self.cuda = cuda
        # self.input_len = input_len
        # self.output_size = output_size
        # self.linear_layer = nn.Linear(self.input_len,self.output_size)
        # self.dropout = nn.Dropout(p=dropout_rate)
        # #print('input_len type: ',type(input_len))
        # #print('output_size type : ',type(output_size))
        # self.act = nn.ELU()
    def forward(self,dim_embs, target):
        target_norm = target.div(torch.norm(target, p=2, dim=-1, keepdim=True))
        # temp = torch.mul(dim_embs[0].div(torch.norm(dim_embs[0], p=2, dim=-1, keepdim=True)), target_norm).sum(1).unsqueeze(1)
        emb = torch.stack([F.normalize(torch.mul(emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True)), target_norm).sum(1).unsqueeze(1), dim = 0, p = 1) * emb for emb in dim_embs]).sum(0)
        # emb = torch.cat(dim_embs,1)
        # emb_combine = self.linear_layer(emb)
        # emb_combine_act = self.act(emb_combine)
        #emb_combine_act_drop = self.dropout(emb_combine_act)

        return emb


class Combine_Attention2(nn.Module):
    """
    Combine embeddings from different dimensions to generate a general embedding
    """
    def __init__(self, input_len=6, output_size = 3,cuda=False, dropout_rate=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(Combine_Attention2,self).__init__()
        self.shared_MLP = nn.Linear(input_len, output_size)
        self.weight_MLP = nn.Linear(output_size, 1)
        # self.cuda = cuda
        # self.input_len = input_len
        # self.output_size = output_size
        # self.linear_layer = nn.Linear(self.input_len,self.output_size)
        # self.dropout = nn.Dropout(p=dropout_rate)
        # #print('input_len type: ',type(input_len))
        # #print('output_size type : ',type(output_size))
        # self.act = nn.ELU()
    def forward(self,dim_embs, target):
        weight = []
        for i in range(0, len(dim_embs)):
            shared = F.relu(self.shared_MLP(torch.cat((dim_embs[i], target), dim=1)))
            weight.append(self.weight_MLP(shared).squeeze())

        weight = torch.stack(weight).transpose(0, 1)
        weight = F.softmax(weight, dim=1)

        temp = torch.zeros(dim_embs[0].shape[0], dim_embs[0].shape[1]).to(dim_embs[0])
        for i in range(0, len(dim_embs)):
            temp += dim_embs[i] * weight[:, i].unsqueeze(-1)

        return temp

class mGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, weight, views, alpha, dropout=0.5, self_loop=True):
        super(mGCN, self).__init__()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.attentions1 = Attention(in_channels)
        self.attentions2 = Attention(hidden_channels)
        self.comb1 = Combine(hidden_channels* views, hidden_channels)
        self.comb2 = Combine(hidden_channels * views, hidden_channels)
        self.self_loop = self_loop
        self.act = nn.ELU()
        self.alpha = alpha

        for i in range(0, views):
            self.conv1.append(GCNConv(in_channels, hidden_channels))

        for i in range(0, views):
            self.conv2.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(p = dropout)
        self.weight = weight
        self.views = views

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def get_cross_rep(self, x_inner, x, attentions, last=True):
        x_cross = []
        for i in range(0, len(x)):
            temp = torch.zeros(x[i].shape[0], x[i].shape[1]).to(x[i])
            for j in range(len(x)):
                if self.self_loop:
                    temp = temp + self.alpha*attentions[i,j]*x[j]
                else:
                    if j!=i:
                        temp = temp + self.alpha*attentions[i,j]*x[j]
            x_cross.append(temp)
        if last:
            x_res = [(1-self.alpha) + emb_dim_inner + self.dropout(self.act(emb_dim)) for emb_dim, emb_dim_inner in zip(x_cross, x_inner)]
        else:
            x_res = [(1 - self.alpha) + emb_dim_inner + emb_dim for emb_dim, emb_dim_inner in zip(x_cross, x_inner)]

        return x_res

    def forward(self, data, edges, edges_neg, test_view=None):
        x, edge_index = data.x, data.edge_index
        x_multi = []
        Ws = []
        x_lin = []
        for i, conv in enumerate(self.conv1):
            x_temp = self.act(conv(x, edge_index[i]))
            x_temp = self.dropout(x_temp)
            x_multi.append(x_temp)
            x_lin.append(conv.lin(x))
            Ws.append(conv.lin.weight)

        attentions = self.attentions1(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions)
        x = self.comb1(x_multi)

        x_lin = []
        x_multi = []
        Ws = []
        for i, conv in enumerate(self.conv2):
            x_temp = conv(x, edge_index[i])
            x_multi.append(x_temp)
            x_lin.append(conv.lin(x))
            Ws.append(conv.lin.weight)

        attentions = self.attentions2(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions, last=True)
        # if self.weight:
        #     weight = data.graph['weight']
        #     x = F.relu(self.conv1(x, edge_index, weight))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.conv2(x, edge_index, weight)
        # else:
        #     # x = self.linear(x)
        #     # x = self.conv2(x, edge_index)
        #     x = F.relu(self.conv1(x, edge_index))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.conv2(x, edge_index)
        if test_view is None:
            results = []
            for i in range(0, self.views):
                x = x_multi[i]
                result = torch.cat(((x[edges[i][:, 0]] * x[edges[i][:, 1]]).sum(dim=-1),
                                     (x[edges_neg[i][:, 0]] * x[edges_neg[i][:, 1]]).sum(dim=-1)))
                results.append(result)
        else:
            x = x_multi[test_view]
            
            results = torch.cat(((x[edges[:, 0]]* x[edges[:, 1]]).sum(dim=-1),
                                 (x[edges_neg[:, 0]]* x[edges_neg[:, 1]]).sum(dim=-1)))

        return results

