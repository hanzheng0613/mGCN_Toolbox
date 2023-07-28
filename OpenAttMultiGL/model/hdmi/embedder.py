"""
   PyTorch implementation of HDMI: High-order Deep Multiplex Infomax  
   
       https://github.com/baoyujing/HDMI/tree/master
        
"""

import torch
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import *

class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        self.sample_data = dataset(args.dataset)
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":

            features_list,features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,train_id,valid_id,test_id = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "acm":
            
            features_list,features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list, sequence_adj,train_id,valid_id,test_id = self.sample_data.load_data(args.dataset)
            
            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "imdb":
            
            features_list,features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list, sequence_adj,train_id,valid_id,test_id = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        if args.dataset == "amazon":
            
            features_list,features, data_set, num_dims, num_classes, gcn_labels,labels, gcn_adj_list, adj_list, edge_list, sequence_adj,train_id,valid_id,test_id = self.sample_data.load_data(args.dataset)

            idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        c = args.test_view
        neg_num = 1
        for i in edge_list:
            i = i.transpose(1, 0)
            
        split_edges = mask_test_edges(features, edge_list[c],neg_num)
        preprocessed_features = preprocess_features(features)

        args.nb_nodes = adj_list[0].shape[0]
        args.ft_size = preprocessed_features[0].shape[1]
        args.nb_classes = labels.shape[0]

        gcn_adj_list = [normalize_adj(adj) for adj in gcn_adj_list]
        self.adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in gcn_adj_list]
        self.features = torch.FloatTensor(preprocessed_features)
        self.labels = torch.FloatTensor(gcn_labels).to(args.device)
        self.idx_train = torch.LongTensor(train_id).to(args.device)
        self.idx_val = torch.LongTensor(valid_id).to(args.device)
        self.idx_test = torch.LongTensor(test_id).to(args.device)
        split_edges['train']['label'] = torch.cat(
            (split_edges['train']['label_pos'], split_edges['train']['label_neg'])).to(args.device)
        split_edges['valid']['label'] = torch.cat(
            (split_edges['valid']['label_pos'], split_edges['valid']['label_neg'])).to(args.device)
        split_edges['test']['label'] = torch.cat(
            (split_edges['test']['label_pos'], split_edges['test']['label_neg'])).to(args.device)
        self.split_edge = split_edges

        self.args = args
