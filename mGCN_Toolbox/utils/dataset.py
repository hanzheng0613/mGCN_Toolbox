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


from mGCN_Toolbox.utils.process import*

from mGCN_Toolbox.visualization.multilayer_graph import LayeredNetworkGraph

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

class dataset():
    
    def __init__(self, dataname): #(self, dataname, train_percent, valid_percent):
        self.dataname = dataname
        #self.gs = None
        #self.HAN_features = None
        #self.HAN_labels = None
        #self.HAN_num_classes = None
        #self.HAN_train_idx = None
        #self.HAN_val_idx = None
        #self.HAN_test_idx = None
        #self.train_mask = None
        #self.val_mask = None
        #self.test_mask = None
        #if self.dataname == 'amazon'or self.dataname == 'acm'or self.dataname == 'imdb' or self.dataname != 'dblp':
        self.truefeatures_list, self.features, self.dataset, self.num_dims, self.num_classes, self.gcn_labels, self.labels, self.gcn_adj_list,self.adj_list, self.edge_list, self.sequence_adj = self.load_data(self.dataname)
        
        #if self.dataname == 'Amazon'or self.dataname == 'Youtube'or self.dataname == 'Twitter'or self.dataname != 'example':
        self.training_data_by_type, self.valid_true_data_by_edge, self.valid_false_data_by_edge, self.testing_true_data_by_edge, self.testing_false_data_by_edge = self.load_txt_data(self.dataname)
        
        self.gs,self.HAN_features,self.HAN_labels,self.HAN_num_classes,self.HAN_train_idx,self.HAN_val_idx,self.HAN_test_idx,self.train_mask,self.val_mask,self.test_mask = self.load_HAN_data(self.dataname)
        
        
        self.X, self.av, self.gnd = self.load_MvAGC_data(self.dataname)
            
                                           #self.gs,self.HAN_features,self.HAN_labels,self.HAN_num_classes,self.HAN_train_idx,self.HAN_val_idx,
                #self.HAN_test_idx,self.train_mask,self.val_mask,self.test_mask = self.load_HAN_data(self.dataname)
        
        

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
            return truefeatures_list, truefeatures, dataset, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list,sequence_adj
        
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
        
        return truefeatures_list, features, dataset, num_dims, num_classes, gcn_labels, labels, gcn_adj_list, adj_list, edge_list,sequence_adj
    
    
    def load_txt_data(self,dataname):
        if self.dataname == 'Amazon'or self.dataname == 'Youtube'or self.dataname == 'Twitter'or self.dataname == 'example':
            training_data_by_type = load_txt_training_data("mGCN_Toolbox/data/GATNE/"+self.dataname + "/train.txt")
            valid_true_data_by_edge, valid_false_data_by_edge = load_txt_testing_data("mGCN_Toolbox/data/GATNE/"+self.dataname + "/valid.txt")
            testing_true_data_by_edge, testing_false_data_by_edge = load_txt_testing_data(
            "mGCN_Toolbox/data/GATNE/"+ self.dataname + "/test.txt")
            
        else: 
            training_data_by_type = None
            valid_true_data_by_edge = None
            valid_false_data_by_edge = None
            testing_true_data_by_edge = None
            testing_false_data_by_edge = None
            
        return training_data_by_type, valid_true_data_by_edge, valid_false_data_by_edge, testing_true_data_by_edge, testing_false_data_by_edge
    
    def load_HAN_data(self, dataname):
        if self.dataname == 'AMAZON':
            gs,HAN_features,HAN_labels,HAN_num_classes,HAN_train_idx,HAN_val_idx,HAN_test_idx,train_mask,val_mask,test_mask = load_AMAZON(False)

        elif self.dataname == 'ACM':
            gs,HAN_features,HAN_labels,HAN_num_classes,HAN_train_idx,HAN_val_idx,HAN_test_idx,train_mask,val_mask,test_mask = load_ACM(False)

        elif self.dataname == 'IMDB':
            gs,HAN_features,HAN_labels,HAN_num_classes,HAN_train_idx,HAN_val_idx,HAN_test_idx,train_mask,val_mask,test_mask = load_IMDB(False)

        elif self.dataname == 'ACM_RAW':
            gs,HAN_features,HAN_labels,HAN_num_classes,HAN_train_idx,HAN_val_idx,HAN_test_idx,train_mask,val_mask,test_mask = load_ACM_raw(False)
            
        
        else:
            gs = None
            HAN_features = None
            HAN_labels = None
            HAN_num_classes = None
            HAN_train_idx = None
            HAN_val_idx = None
            HAN_test_idx = None
            train_mask = None
            val_mask = None
            test_mask = None
        return gs,HAN_features,HAN_labels,HAN_num_classes,HAN_train_idx,HAN_val_idx,HAN_test_idx,train_mask,val_mask,test_mask
    
    def load_MvAGC_data(self,dataname):
        
        #if(dataname == 'large_cora'):
            #X = data['X']
            #A = data['G']
            #gnd = data['labels']
            #gnd = gnd[0, :]
        if dataname == 'ACM3025':
            data = sio.loadmat('mGCN_Toolbox/data/MvAGC/{}.mat'.format(dataname))
            X = data['feature']
            A = data['PAP']
            B = data['PLP']
            av=[]
            av.append(A)
            av.append(B)
            gnd = data['label']
            gnd = gnd.T
            gnd = np.argmax(gnd, axis=0)
        else:
            X= None
            av = None
            gnd = None

        return X, av, gnd


    
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