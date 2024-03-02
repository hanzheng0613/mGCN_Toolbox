import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import * 
from OpenAttMultiGL.model.GATNE.utils import *
#from mGCN_Toolbox.model.GATNE.walk import *
from OpenAttMultiGL.model.GATNE.embedder_link import *
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter

#from utils import *


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class GATNEModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    ):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a

        self.features = None
        if features is not None:
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = Parameter(torch.FloatTensor(feature_dim, embedding_size))
            self.u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))
        else:
            self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
            self.node_type_embeddings = Parameter(
                torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
            )
        self.trans_weights = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            self.embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
            self.u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        else:
            self.node_embeddings.data.uniform_(-1.0, 1.0)
            self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        if self.features is None:
            node_embed = self.node_embeddings[train_inputs]
            node_embed_neighbors = self.node_type_embeddings[node_neigh]
        else:
            node_embed = torch.mm(self.features[train_inputs], self.embed_trans)
            node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)
        node_embed_tmp = torch.diagonal(node_embed_neighbors, dim1=1, dim2=3).permute(0, 3, 1, 2)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)

        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n



    

if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    #print(args)
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None
    
    s = dataset('dblp')
    num_nodes = s.sequence_adj[0].shape[0]
    #print('node:', s.sequence_adj[0].shape[0])
    
    edge_type_count = len(s.sequence_adj)
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    
    dim_a = args.att_dim
    features = None
    
    num_sampled = args.negative_samples
    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)
    #print(type(model.node_embeddings))
    #model.to(device)
    #nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4
    )
    
    num_classes = s.gcn_labels.shape[1]
    split_edges = mask_test_edges(s.features, s.edge_list[0],1)
    
    split_edges['train']['label'] = torch.cat(
        (split_edges['train']['label_pos'], split_edges['train']['label_neg']))#.to(args.device)
    split_edges['valid']['label'] = torch.cat(
        (split_edges['valid']['label_pos'], split_edges['valid']['label_neg']))#.to(args.device)
    split_edges['test']['label'] = torch.cat(
        (split_edges['test']['label_pos'], split_edges['test']['label_neg']))#.to(args.device)
    s_edge = split_edges
    #split_edge = mask_test_edges(t.HAN_features, split_edges, 1, 0.1, 0.5)
    #model.node_embeddings.detach().numpy()
    AUC, ap, hits = link_evaluate(model.node_embeddings.detach().numpy(),s_edge,num_classes)
    print("Average-percision:", np.mean(ap), np.std(ap))
    print("Average-AUC:", np.mean(AUC), np.std(AUC))


