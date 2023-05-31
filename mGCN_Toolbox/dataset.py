import scipy.sparse as sp
import numpy as np
import torch
import pickle as pkl
import scipy.io as sio


from process import*

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

class Dataset():
    
    def __init__(self, dataname, train_percent, valid_percent):
        self.dataname = dataname
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.dataset, self.num_dims, self.training_id, self.valid_id, self.test_id, self.num_classes, self.labels, self.adj_list, self.edge_list = self.load_data(self.dataname, self.train_percent, self.valid_percent)
        
    def split_data(self, node_num):
        
        ind = np.arange(0, node_num, 1)
        training_sample = int(node_num * self.train_percent)
        valid_sample = int(node_num * self.valid_percent)
        np.random.shuffle(ind)
        training_ind = ind[:training_sample]
        valid_id = ind[training_sample:training_sample + valid_sample]
        test_id = ind[training_sample + valid_sample:]
        training_id = torch.LongTensor(training_ind)
        valid_id = torch.LongTensor(valid_id)
        test_id = torch.LongTensor(test_id)
        
        return training_id, valid_id, test_id

    def load_data(self, dataname, training_percent, valid_precent):

        if self.dataname == 'amazon':
            data_filename = './data/amazon/amazon.pkl'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_amazon(data_filename)
        elif self.dataname == 'acm':
            data_filename = './data/acm/acm.mat'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_acm_mat()
        elif self.dataname == 'imdb':
            data_filename = './data/imdb/imdb.pkl'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_imdb(data_filename)
        elif self.dataname == 'dblp':
            data_filename = './data/dblp/dblp.pkl'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_dblp(data_filename)
        embed_matrix = preprocess_features(embed_matrix)
        adj_list = [normalize_adj(adj) for adj in adj_list]
        adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        labels =  np.argmax(labels, axis=1)
        training_id = torch.LongTensor(idx_train)
        valid_id = torch.LongTensor(idx_val)
        test_id = torch.LongTensor(idx_test)

        # edge_list = [add_self_loops(torch.LongTensor(temp).transpose(0, 1))[0] for temp in edge_list]
        edge_list = [torch.LongTensor(temp).transpose(0, 1) for temp in edge_list]

        embed_matrix = torch.FloatTensor(embed_matrix)
        dataset = Data(x=embed_matrix, edge_index=edge_list)
        
        num_classes = np.max(labels) + 1
        labels = torch.LongTensor(labels)
        
        training_id, valid_id, test_id = self.split_data(len(labels))

        return dataset, num_dims, training_id, valid_id, test_id, num_classes, labels, adj_list, edge_list