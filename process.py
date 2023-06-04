import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp

from torch_geometric.utils import from_scipy_sparse_matrix


def load_acm_mat():
    data = sio.loadmat('mGCN_Toolbox/data/acm/acm.mat')
    label = data['label']
    

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*3
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]
    
    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list



def load_dblp():
    data = pkl.load(open("mGCN_Toolbox/data/dblp/dblp.pkl", "rb"))
    label = data['label']

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*3
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*3
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_imdb():
    data = pkl.load(open("mGCN_Toolbox/data/imdb/imdb.pkl", "rb"))
    label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*3
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_amazon():
    data = pkl.load(open("mGCN_Toolbox/data/amazon/amazon.pkl", "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*3
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*3
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


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

def split_data(self, node_num, train_percent, valid_percent):
        
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


if __name__ == '__main__':
    load_acm_mat(sc=3)