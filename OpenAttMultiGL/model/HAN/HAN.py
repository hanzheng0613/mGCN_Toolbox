import torch.nn as nn
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch import Tensor
from typing import Union
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.utils import to_undirected


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        #print('z',z.shape)
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)

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





class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

    
class HAN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout, alpha,self_loop = True
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.attentions1 = SemanticAttention(in_size)
        self.attentions2 = SemanticAttention(hidden_size)
        self.comb1 = Combine(hidden_size, hidden_size)
        self.comb2 = Combine(hidden_size* num_heads[0], hidden_size)
        #self.comb2 = Combine(hidden_size* num_heads[0], hidden_size)
        self.dropout = nn.Dropout(p = dropout)
        self.self_loop = self_loop
        self.act = nn.ELU()
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.alpha = alpha
        for i in range(0, 1):
            self.conv1.append(GCNConv(in_size,hidden_size))

        for i in range(0, 1):
            self.conv2.append(GCNConv(hidden_size,hidden_size))
        
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
        #print('ws',Ws)
        Ws = torch.stack(Ws)
        #for i in Ws:
           # i = i.detach().numpy()
        #print('ws',Ws)
        #Ws = torch.as_tensor(Ws)
        attentions = self.attentions1(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions)
        #print('x',x_multi.shape)
        x = self.comb1(x_multi)

        x_lin = []
        x_multi = []
        Ws = []
        for i, conv in enumerate(self.conv2):
            #print('x',x.shape)
            #print('edge', edge_index[i].shape)
            x_temp = conv(x, edge_index[i])
            x_multi.append(x_temp)
            x_lin.append(conv.lin(x))
            Ws.append(conv.lin.weight)
        #Ws = torch.FloatTensor(Ws)
        Ws = torch.stack(Ws)
        #Ws = torch.as_tensor(Ws)
        #print('ws',Ws.size())
        attentions = self.attentions2(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions, last=True)
        
        results = []
        if test_view is None:
            
            for i in range(0, 1):
                x = x_multi[i]
                result = torch.cat(((x[edges[i][:, 0]] * x[edges[i][:, 1]]).sum(dim=-1),
                                     (x[edges_neg[i][:, 0]] * x[edges_neg[i][:, 1]]).sum(dim=-1)))
                results.append(result)
        else:
            x = x_multi[test_view]
            #print(type(x))
            #print(x)
            #results = torch.cat(((x[edges[:, 0]]).sum(dim=-1),
                                 #(x[edges_neg[:, 0]]).sum(dim=-1)))
            results = torch.cat(((x[edges[:, 0]]* x[edges[:, 1]]).sum(dim=-1),
                                 (x[edges_neg[:, 0]]* x[edges_neg[:, 1]]).sum(dim=-1)))
        #print(results)
        return results


