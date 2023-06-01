import scipy.sparse as sp
import numpy as np
import torch
import pickle as pkl
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

import scipy.io as sio
import scipy.sparse as sp


import holoviews as hv


from process import*

from visualization.multilayer_graph import LayeredNetworkGraph

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

class dataset():
    
    def __init__(self, dataname): #(self, dataname, train_percent, valid_percent):
        self.dataname = dataname
        self.dataset, self.num_dims,  self.num_classes, self.labels, self.adj_list, self.edge_list = self.load_data(self.dataname)
        
        self.visualization = self.visualization(self.dataname)

        #self.dataset, self.num_dims, self.training_id, self.valid_id, self.test_id, self.num_classes, self.labels, #self.adj_list, self.edge_list = self.load_data(self.dataname, self.train_percent, self.valid_percent)
        
    #def split_data(self, node_num, train_percent, valid_percent):
        
        #ind = np.arange(0, node_num, 1)
        #training_sample = int(node_num * self.train_percent)
        #valid_sample = int(node_num * self.valid_percent)
        #np.random.shuffle(ind)
        #training_ind = ind[:training_sample]
        #valid_id = ind[training_sample:training_sample + valid_sample]
        #test_id = ind[training_sample + valid_sample:]
        #training_id = torch.LongTensor(training_ind)
        #valid_id = torch.LongTensor(valid_id)
        #test_id = torch.LongTensor(test_id)
        
        #return training_id, valid_id, test_id

    def load_data(self, dataname):

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
        
        #training_id, valid_id, test_id = self.split_data(len(labels))

        #return dataset, num_dims, training_id, valid_id, test_id, num_classes, labels, adj_list, edge_list
        
        return dataset, num_dims, num_classes, labels, adj_list, edge_list
    
    def visualization(self,dataname):
        if self.dataname == 'amazon':
            data_filename = './data/amazon/amazon.pkl'
            truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list = load_amazon()
        elif self.dataname == 'acm':
            data_filename = './data/acm/acm.mat'
            truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list = load_acm_mat()
        elif self.dataname == 'imdb':
            data_filename = './data/imdb/imdb.pkl'
            truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list = load_imdb()
        elif self.dataname == 'dblp':
            data_filename = './data/dblp/dblp.pkl'
            truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list = load_dblp()
            
        
        graph_list = []
        
        for i in adj_list:
            i = i[0:100,0:100]
            G = nx.from_scipy_sparse_array(i)
            graph_list.append(G)
            
        
        
        self.features = truefeatures[0:100,0:100]
        self.attribute = preprocess_features(self.features)
        
        LayeredNetworkGraph(graph_list,graphs_attribute=self.attribute,layout=nx.random_layout)