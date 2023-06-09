'''
    The PyTorch implementation of Unsupervised Attributed Multiplex Network Embedding (DMGI)
    
    https://github.com/pcy1302/DMGI
'''

import time
import numpy as np
import torch

import torch.nn as nn

from mGCN_Toolbox.utils.dataset import dataset
from mGCN_Toolbox.utils.process import *
from mGCN_Toolbox.layers.DMGI import AvgReadout


class embedder:
    def __init__(self, args):
        self.sample_data = dataset(args.dataset)
        args.batch_size = 1
        args.sparse = True
        args.metapaths_list = args.metapaths.split(",")
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        truefeatures_list, features, data_set, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list, sequence_adj = self.sample_data.load_data(args.dataset)
        idx_train, idx_val, idx_test = split_node_data(len(self.sample_data.labels),args.training_ratio,args.validing_ratio)
        
        #adj, features, labels, idx_train, idx_val, idx_test = process.load_data_dblp(args)
        
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
