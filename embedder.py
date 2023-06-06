"""
   PyTorch implementation of HDMI: High-order Deep Multiplex Infomax  
   
       https://github.com/baoyujing/HDMI/tree/master
        
"""

import torch
from mGCN_Toolbox.utils.dataset import dataset
from mGCN_Toolbox.utils.process import *

class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        self.sample_data = dataset(args.dataset)
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":

            features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "acm":
            
            features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list = self.sample_data.load_data(args.dataset)
            
            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "imdb":
            
            features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "amazon":
            
            features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)

        preprocessed_features = preprocess_features(features)

        args.nb_nodes = adj_list[0].shape[0]
        args.ft_size = preprocessed_features[0].shape[1]
        args.nb_classes = labels.shape[0]

        gcn_adj_list = [normalize_adj(adj) for adj in gcn_adj_list]
        self.adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in gcn_adj_list]
        self.features = torch.FloatTensor(preprocessed_features)
        self.labels = torch.FloatTensor(gcn_labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args
