'''
     The PyTorch implementation of X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning\
     
     https://github.com/baoyujing/X-GOAL/tree/main    
'''


import os

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import *
from OpenAttMultiGL.layers.X_GOAL import TransformationPOS, TransformationNEG

from OpenAttMultiGL.model.X_GOAL.utils import preprocess_features_dblp


class Model:
    def __init__(self, args):
        self.args = args
        self.sample_data = dataset(args.dataset)
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":
            features_list,features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,idx_train, idx_val, idx_test = self.sample_data.load_data(args.dataset)
            #idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
            
            #adj_list, features, labels, idx_train, idx_val, idx_test = load_dblp()
            
        if args.dataset == "acm":
            features_list,features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,idx_train, idx_val, idx_test = self.sample_data.load_data(args.dataset)
            #idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
            #adj_list, features, labels, idx_train, idx_val, idx_test = load_acm_mat()
            
        if args.dataset == "imdb":            
            features_list,features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,idx_train, idx_val, idx_test = self.sample_data.load_data(args.dataset)
            #idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
            #adj_list, features, labels, idx_train, idx_val, idx_test = load_imdb()
            
        if args.dataset == "amazon":
            features_list,features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,idx_train, idx_val, idx_test = self.sample_data.load_data(args.dataset)
            #idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
            
            #adj_list, features, labels, idx_train, idx_val, idx_test = load_amazon()
        args_test_view = 0
        c = args_test_view
        neg_num = 1
        split_edges = mask_test_edges(features, edge_list[c],neg_num)
        
        split_edges['train']['label'] = torch.cat(
            (split_edges['train']['label_pos'], split_edges['train']['label_neg'])).to(args.device)
        split_edges['valid']['label'] = torch.cat(
            (split_edges['valid']['label_pos'], split_edges['valid']['label_neg'])).to(args.device)
        split_edges['test']['label'] = torch.cat(
            (split_edges['test']['label_pos'], split_edges['test']['label_neg'])).to(args.device)
        self.split_edge = split_edges
        
        
        if args.dataset == "dblp":
            features = preprocess_features_dblp(features)
        else:
            features = preprocess_features(features)

        self.args.nb_nodes = adj_list[0].shape[0]
        self.args.ft_size = features.shape[1]
        self.args.nb_classes = gcn_labels.shape[1]

        normalized_sequence_adj = [sequence_normalize_adj(adj) for adj in sequence_adj]
        self.adj_list = [torch.FloatTensor(adj) for adj in normalized_sequence_adj]
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(gcn_labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args
        self.criteria = nn.CrossEntropyLoss()

        self.transform_pos = TransformationPOS(p_drop=self.args.p_drop)
        self.transform_neg = TransformationNEG()

        self.features = self.features.to(self.args.device)
        self.adj_list = [adj.to(args.device) for adj in self.adj_list]
        self.n_adj = len(self.adj_list)

        self.cos = nn.CosineSimilarity()

    def evaluate(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def warmup(self):
        raise NotImplemented

    def _train_full_loss(self):
        raise NotImplemented

    def info_nce(self, h_org, h_pos, h_neg):
        score_pos = self.cos(h_org, h_pos)
        score_neg = self.cos(h_org, h_neg)

        score_pos = torch.exp(score_pos)
        score_neg = torch.exp(score_neg)

        score_all = score_pos + score_neg
        loss = torch.log(score_pos / score_all)
        loss = -torch.mean(loss)
        return loss

    def node_alignment(self, h_list, h_list_neg):
        n = len(h_list)
        score_pos_list = []
        score_neg_list = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                score_pos = self.cos(h_list[i], h_list[j])
                score_neg = self.cos(h_list[i], h_list_neg[i])
                score_pos_list.append(score_pos)
                score_neg_list.append(score_neg)
        score_pos = torch.stack(score_pos_list, dim=0)
        score_neg = torch.stack(score_neg_list, dim=0)
        score_pos = torch.exp(score_pos)
        score_neg = torch.exp(score_neg)
        score_all = score_pos + score_neg
        loss = torch.log(score_pos / score_all)
        loss = -torch.mean(loss)
        return loss

    def run_kmeans(self, x, k):
        x_np = x.detach().cpu().data.numpy()
        kmeans = KMeans(n_clusters=k,n_init = 10) #n_jobs=16)
        kmeans.fit(x_np)
        c_cond_x = kmeans.predict(x_np)  # [batch_size,]
        centroids = kmeans.cluster_centers_  # [k, dim]

        centroids = torch.transpose(torch.from_numpy(centroids), 0, 1).to(x.device)  # [k, dim] -> [dim, k]
        target = torch.from_numpy(c_cond_x).type(torch.LongTensor).to(x.device)
        return centroids, target

    def loss_kmeans(self, x, centroids, tau, target):
        logits = torch.matmul(x, centroids)  # [batch_size, k]
        logits = logits/tau
        loss = self.criteria(logits, target)
        return loss
