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


from OpenAttMultiGL.utils.process import*

from OpenAttMultiGL.visualization.multilayer_graph import LayeredNetworkGraph

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

class dataset():
    
    def __init__(self, dataname): #(self, dataname, train_percent, valid_percent):
        self.dataname = dataname
        
        self.truefeatures_list, self.features, self.dataset, self.num_dims, self.num_classes, self.gcn_labels, self.labels, self.gcn_adj_list,self.adj_list, self.edge_list, self.sequence_adj,self.train_id,self.valid_id,self.test_id = self.load_data(self.dataname)
        
       
        

    def load_data(self, dataname):
        if self.dataname == 'amazon':
            data_filename = './data/amazon/amazon.pkl'
            truefeatures_list, truefeatures, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_amazon(3)
        elif self.dataname == 'acm':
            data_filename = './data/acm/acm.mat'
            truefeatures_list, truefeatures, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_acm_mat(3)
        elif self.dataname == 'imdb':
            data_filename = './data/imdb/imdb.pkl'
            truefeatures_list, truefeatures, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_imdb(3)
        elif self.dataname == 'dblp':
            data_filename = './data/dblp/dblp.pkl'
            truefeatures_list, truefeatures, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_dblp(3)
        #elif: self.dataname == 'amazon'|| self.dataname == 'acm'|| self.dataname == 'imdb'||self.dataname != 'dblp'
        else:
            truefeatures_list = None
            truefeatures = None
            dataset = None
            num_dims = None
            num_classes = None
            gcn_labels = None
            labels = None
            gcn_adj_list = None
            adj_list = None
            edge_list = None
            sequence_adj = None
            idx_train = None
            idx_val = None
            idx_test = None
            return truefeatures_list, truefeatures, dataset, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list,sequence_adj,idx_train, idx_val, idx_test
        
        features = truefeatures
        truefeatures = preprocess_features(truefeatures)
        gcn_labels = labels
        gcn_adj_list = adj_list
        adj_list = [normalize_adj(adj) for adj in adj_list]
        adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        labels =  np.argmax(labels, axis=1)
        training_id = torch.LongTensor(idx_train)
        valid_id = torch.LongTensor(idx_val)
        test_id = torch.LongTensor(idx_test)

        
        edge_list = [torch.LongTensor(temp).transpose(0, 1) for temp in edge_list]

        truefeatures = torch.FloatTensor(truefeatures)
        dataset = Data(x=truefeatures, edge_index=edge_list)
        
        num_classes = np.max(labels) + 1
        labels = torch.LongTensor(labels)
        
        #training_id, valid_id, test_id = self.split_data(len(labels))

        #return dataset, num_dims, training_id, valid_id, test_id, num_classes, labels, adj_list, edge_list
        
        return truefeatures_list, features, dataset, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list,sequence_adj,training_id,valid_id,test_id
    
    
    
    
   
    
    def visualization(self,dataname):
        if self.dataname == 'amazon':
            data_filename = './data/amazon/amazon.pkl'
            truefeatures_list, truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_amazon()
        elif self.dataname == 'acm':
            data_filename = './data/acm/acm.mat'
            truefeatures_list, truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_acm_mat()
        elif self.dataname == 'imdb':
            data_filename = './data/imdb/imdb.pkl'
            truefeatures_list, truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_imdb()
        elif self.dataname == 'dblp':
            data_filename = './data/dblp/dblp.pkl'
            truefeatures_list, truefeatures, edge_index, len_edge_index, label, idx_train, idx_val, idx_test, adj_list,sequence_adj = load_dblp()
            
        
        graph_list = []
        
        for i in adj_list:
            i = i[0:100,0:100]
            G = nx.from_scipy_sparse_array(i)
            graph_list.append(G)
            
        
        
        self.features = truefeatures[0:100,0:100]
        self.attribute = preprocess_features(self.features)
        
        LayeredNetworkGraph(graph_list,graphs_attribute=self.attribute,layout=nx.random_layout)