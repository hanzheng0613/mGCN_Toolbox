import math
import argparse

import numpy as np
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import * 
from OpenAttMultiGL.layers.hdmi.gcn import GCN
from OpenAttMultiGL.model.GATNE.utils import *
from OpenAttMultiGL.model.GATNE.evaluate import evaluate
#from mGCN_Toolbox.model.GATNE.walk import *
#from OpenAttMultiGL.model.X_GOAL.evaluate import *

def combine_att(h_list):
    att_act1 = nn.Tanh()
    att_act2 = nn.Softmax(dim=-1)
    h_combine_list = []
    for i, h in enumerate(h_list):
        h = w_list[i](h)
        h = y_list[i](h)
        h_combine_list.append(h)
    score = torch.cat(h_combine_list, -1)
    score = att_act1(score)
    score = att_act2(score)
    score = torch.unsqueeze(score, -1)
    h = torch.stack(h_list, dim=1)
    h = score * h
    h = torch.sum(h, dim=1)
    return h

def embed(seq, adj_list, sparse):
    global w_list
    global y_list
    gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
    w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
    y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])
    h_1_list = []
    for i, adj in enumerate(adj_list):
        h_1 = torch.squeeze(gcn_list[i](seq, adj, sparse))
        h_1_list.append(h_1)
    h = combine_att(h_1_list)
    return h.detach()





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


def train_model(network_data, feature_dic,dataset):
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    features = None
    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        features = torch.FloatTensor(features).to(device)

    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4
    )
    best_micro = 0
    best_macro = 0
    best_score = 0
    test_score = (0.0, 0.0, 0.0)
    patience = 0
    sparse = True
    labels = torch.FloatTensor(dataset.gcn_labels)
    idx_train = torch.LongTensor(dataset.train_id)
    idx_val = torch.LongTensor(dataset.valid_id)
    idx_test = torch.LongTensor(dataset.test_id)
    macro = []
    micro = []
    k1_list = []
    sim_list = []
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)

        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0
        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device),)
            loss = nsloss(data[0].to(device), embs, data[1].to(device))
            loss.backward()
            optimizer.step()
            #print('embs: ', embs)
            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))
        
        model.eval()
        features = torch.FloatTensor(preprocessed_features)
        gcn_adj_list = [normalize_adj(adj) for adj in dataset.gcn_adj_list]
        adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in gcn_adj_list]
        embeds = embed(features, adj_list, sparse)
        
        macro_f1s, micro_f1s, k1, sim = evaluate(embeds, idx_train, idx_val, idx_test, labels)
        f1_macro = np.mean(macro_f1s)
        f1_micro = np.mean(micro_f1s)
        
        macro.append(f1_macro)
        micro.append(f1_micro)
        k1_list.append(k1)
        sim_list.append(sim)
    #return average_micro,average_macro,average_sim,average_nmi
    return macro,micro,k1_list,sim_list


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    t = dataset('imdb')
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None
        
    preprocessed_features = preprocess_features(t.features)
    ft_size = preprocessed_features[0].shape[1] 
    hid_units = 128
    n_networks = len(t.adj_list)
    
    #embeds = embed(features, adj_list, self.args.sparse)
    # Write down data in format required for model training
    #f = open("OpenAttMultiGL/data/GATNE/Amazon/testt.txt","a")
    #d = dict()
    #for i in range(len(t.sequence_adj)):
        #d[i] = []
        #for j in range(len(t.sequence_adj[i])):
            #for l in t.test_id:
                #if j == l:
                    #for k in range(len(t.sequence_adj[i][j])):
                        #f.write(str(i))
                        #f.write(' ')
                        #f.write(str(j))
                        #f.write(' ')
                        #f.write(str(k))
                        #f.write(' ')
                        #f.write(str(int(t.sequence_adj[i][j][k])))
                        #f.write('\n')
    #f.close()
    #training_data_by_type = load_training_data("OpenAttMultiGL/data/GATNE/"+file_name + "/train.txt")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        "OpenAttMultiGL/data/GATNE/"+file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        "OpenAttMultiGL/data/GATNE/"+file_name + "/test.txt"
    )
    
    #c = t.sequence_adj[0][]
    d = dict()
    for i in range(len(t.sequence_adj)):
        d[i] = []
        for j in range(len(t.sequence_adj[i])):
            for l in t.train_id:
                if j == l:
                    for k in range(len(t.sequence_adj[i][j])):
                        if t.sequence_adj[i][j][k] == 1:
                            e = (str(j),str(k))
                            d[i].append(e)
    #micro,macro,sim,nmi = train_model(d, feature_dic,t)
    micro,macro,nmi,sim = train_model(d, feature_dic,t)
    
    print("Final score: \n")
    print('Micro: {:.4f} ({:.4f})'.format(np.mean(micro),np.std(micro)))
    print('Macro: {:.4f} ({:.4f})'.format(np.mean(macro),np.std(macro)))
    print('Sim: {:.4f} ({:.4f})'.format(np.mean(sim),np.std(sim)))
    print('NMI: {:.4f} ({:.4f})'.format(np.mean(nmi),np.std(nmi)))
    #print('SIM: ', sim)
    #print('NMI: ', nmi)



