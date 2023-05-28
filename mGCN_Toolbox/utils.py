import scipy.sparse as sp
import numpy as np
import torch
import pickle as pkl
import scipy.io as sio
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

def readinDBLP_conf(file):

    dblp_conf_data = np.loadtxt(file,int)
    num_nodes = np.max(dblp_conf_data)+1
    num_dimensions = np.max(dblp_conf_data[:,2]) +1

    return dblp_conf_data, num_nodes, num_dimensions

def load_line_emb(file):
    X = np.loadtxt(fname=file,skiprows=1)
    num_nodes = int(np.max(X)) +1
    len_dim = len(X[0,:])
    Y = np.random.normal(scale=0.1,size=(num_nodes,len_dim))
    X_1 = X[:,0]
    X_1 = np.reshape(X_1,-1)
    X_1 = X_1.astype(int)
    Y[X_1,:] = X
   # print(X)
   # print(Y[[0,1,2,3138,2999,876],:])

    return Y[:,1:]

def load_amazon(path, sc=3):
    data = pkl.load(open(path, "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc


    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)
    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list

def load_acm(path, sc=3):
    data = sio.loadmat('./data/acm/acm.mat')
    label = data['label']

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc


    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()
    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list

def load_dblp(path, sc=3):
    data = pkl.load(open(path, "rb"))
    label = data['label']

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc


    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)
    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list

def load_imdb(path, sc=3):
    data = pkl.load(open(path, "rb"))
    label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)
    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dataname, training_percent=0.3, valid_precent=0.1):
    if dataname == 'epinions':
        data_filename = './data/epinions/ep_multi.txt.gz'
        label_file = './data/epinions/epinions_class_info.txt'
        path = './data/epinions/'
        ep_data, num_nodes, num_dims = readinDBLP_conf(data_filename)
        num_nodes = int(num_nodes)
        feature_filename = './data/epinions/emb_ep_64_400.txt'

        print('num_nodes: ', num_nodes)
        print('num_dims: ', num_dims)
        print('feature_filename: ', feature_filename)

        embed_matrix = load_line_emb(feature_filename)

        edge_list = []
        for i in range(0, num_dims):
            edge_list.append([])

        for e in ep_data:
            edge_list[e[2]].append([e[0], e[1]])


        labels = np.loadtxt(fname=label_file, dtype=int)
        labels = labels[:, 1]
        ind = np.arange(0, len(labels), 1)
        training_sample = int(len(labels) * training_percent)
        valid_sample = int(len(labels) * valid_precent)
        np.random.shuffle(ind)
        training_ind = ind[:training_sample]
        valid_id = ind[training_sample:training_sample + valid_sample]
        test_id = ind[training_sample + valid_sample:]
        training_id = torch.LongTensor(training_ind)
        valid_id = torch.LongTensor(valid_id)
        test_id = torch.LongTensor(test_id)
    else:
        if dataname == 'amazon':
            data_filename = './data/amazon/amazon.pkl'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_amazon(data_filename)
        elif dataname == 'acm':
            data_filename = './data/acm/acm.mat'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_acm(data_filename)
        elif dataname == 'imdb':
            data_filename = './data/imdb/imdb.pkl'
            embed_matrix, edge_list, num_dims, labels, idx_train, idx_val, idx_test, adj_list = load_imdb(data_filename)
        elif dataname == 'dblp':
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

    return dataset, num_dims, training_id, valid_id, test_id, num_classes, labels, adj_list
