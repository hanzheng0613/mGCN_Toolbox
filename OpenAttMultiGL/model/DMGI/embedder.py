'''
    The PyTorch implementation of Unsupervised Attributed Multiplex Network Embedding (DMGI)
    
    https://github.com/pcy1302/DMGI
'''

import time
import numpy as np
import torch

import torch.nn as nn

from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import *
from OpenAttMultiGL.layers.DMGI import AvgReadout


class embedder:
    def __init__(self, args):
        self.sample_data = dataset(args.dataset)
        args.batch_size = 1
        args.sparse = True
        #args.metapaths_list = args.metapaths.split(",")
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        truefeatures_list, features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj,idx_train, idx_val, idx_test = self.sample_data.load_data(args.dataset)
        #idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        
        #adj, features, labels, idx_train, idx_val, idx_test = process.load_data_dblp(args)
        
        args_test_view = 0
        c = args_test_view
        neg_num = 1
        for i in edge_list:
            i = i.transpose(1, 0)
        split_edges = mask_test_edges(features, edge_list[c],neg_num )
        
        features = [preprocess_features(feature) for feature in truefeatures_list]
        
        

        args.nb_nodes = features[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = gcn_labels.shape[1]
        args.nb_graphs = len(gcn_adj_list)
        args.adj = gcn_adj_list
        adj = [normalize_adj(adj_) for adj_ in gcn_adj_list]
        self.adj = [sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]

        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]

        self.labels = torch.FloatTensor(gcn_labels[np.newaxis]).to(args.device)
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)
        
        split_edges['train']['label'] = torch.cat(
            (split_edges['train']['label_pos'], split_edges['train']['label_neg'])).to(args.device)
        split_edges['valid']['label'] = torch.cat(
            (split_edges['valid']['label_pos'], split_edges['valid']['label_neg'])).to(args.device)
        split_edges['test']['label'] = torch.cat(
            (split_edges['test']['label_pos'], split_edges['test']['label_neg'])).to(args.device)
        self.split_edge = split_edges

        # How to aggregate
        args.readout_func = AvgReadout()

        # Summary aggregation
        args.readout_act_func = nn.Sigmoid()

        self.args = args

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
