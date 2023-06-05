import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils import to_undirected

def load_acm_mat():
    data = sio.loadmat('data/acm/acm.mat')
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
    data = pkl.load(open("data/dblp/dblp.pkl", "rb"))
    label = data['label']

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*3
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*3
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list


def load_imdb():
    data = pkl.load(open("data/imdb/imdb.pkl", "rb"))
    label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*3
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*3

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


def load_amazon():
    data = pkl.load(open("data/amazon/amazon.pkl", "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*3
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*3
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

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

def get_masked(context, edge_index, R, test_edges):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).numpy()
    edges = edge_index.transpose(0, 1).numpy()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(context.shape[0], context.shape[0]),
                        dtype=np.float32)
    adj_neg = 1 - adj.toarray()
    # number_edges = R
    # matrix = np.triu(attention, k=1)
    matrix = np.multiply(adj_neg, attention)
    matrix = torch.FloatTensor(matrix)
    _, edge_candidate = torch.topk(matrix, k =R, dim=0)
    # right = edge_candidate.transpose(0, 1).numpy()
    left = torch.arange(0, matrix.shape[0], dtype=int)
    left = left.repeat(R, 1).transpose(0, 1).flatten()
    edges = torch.stack([left, edge_candidate.flatten()]).transpose(0, 1)
    # edges = get_edge_candidate(matrix, matrix.shape[0])
    # plot_test(attention[edges[:, 0], edges[:, 1]])
    # adj_mask = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(context.shape[0], context.shape[0]),
    #                     dtype=np.float32).toarray()
    #
    # adj_mask = torch.FloatTensor(adj_mask)
    return edges

def get_edges(feature, edge_index):
    edges = edge_index.transpose(0, 1).numpy()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(feature.shape[0], feature.shape[0]),
                        dtype=np.float32)

    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    # pos_edges = pos_edges[0 : int(0.6 * pos_edges.shape[0]), :]

    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = torch.LongTensor(pos_edges)
    split_edge['train']['edge_neg'] = torch.LongTensor(neg_edges)

    split_edge['train']['label_pos'] = torch.ones(split_edge['train']['edge'].size(0), dtype=torch.float)
    split_edge['train']['label_neg'] = torch.zeros(split_edge['train']['edge'].size(0), dtype=torch.float)

    return split_edge

def intersect2D(a, b):
    return len(np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)]))

def split_node_data(node_num, train_percent, valid_percent):
        
    ind = np.arange(0, node_num, 1)
    training_sample = int(node_num * train_percent)
    valid_sample = int(node_num * valid_percent)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    training_id = torch.LongTensor(training_ind)
    valid_id = torch.LongTensor(valid_id)
    test_id = torch.LongTensor(test_id)
        
    return training_id, valid_id, test_id

def split_link_data(data, test_view, neg_k, multi=False):

    if multi:
        split_edge = []
        views = len(data.edge_index)
        for i in range(0, views):
            print("Views:", to_undirected(data.edge_index[i]).shape)
            if i == test_view:
                temp = mask_test_edges(data.x, data.edge_index[i], neg_k)
                data.edge_index[i] = temp['train']['edge'].t()
            else:
                temp = get_edges(data.x, data.edge_index[i])
            split_edge.append(temp)
    else:
        split_edge = mask_test_edges(data.x, data.edge_index[test_view], neg_k)
        data.edge_index[test_view] = split_edge['train']['edge'].t()

    return data, split_edge

def mask_test_edges(feature, edge_index, neg_num, val_prop=0.1, test_prop=0.5):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    # random.seed(234)
    # torch.manual_seed(234)

    edges = edge_index.transpose(0, 1).numpy()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(feature.shape[0], feature.shape[0]),
                        dtype=np.float32)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)


    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test * neg_num + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    # np.random.shuffle(train_edges_false)
    # if dataname in ['pubmed', 'photo']:
    #     np.random.shuffle(train_edges_false)
    #     train_edges_false = train_edges_false[0: m_pos]
    # train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0).tolist()
    # random.shuffle(train_edges_false)
    # train_edges_false = np.array(train_edges_false)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = torch.LongTensor(train_edges)
    split_edge['train']['edge_neg'] = torch.LongTensor(train_edges_false)
    split_edge['valid']['edge'] = torch.LongTensor(val_edges)
    split_edge['valid']['edge_neg'] = torch.LongTensor(val_edges_false)
    split_edge['test']['edge'] = torch.LongTensor(test_edges)
    split_edge['test']['edge_neg'] = torch.LongTensor(test_edges_false)

    split_edge['train']['label_pos'] = torch.ones(split_edge['train']['edge'].size(0), dtype=torch.float)
    split_edge['train']['label_neg'] = torch.zeros(split_edge['train']['edge'].size(0), dtype=torch.float)

    split_edge['valid']['label_pos'] = torch.ones(split_edge['valid']['edge'].size(0), dtype=torch.float)
    split_edge['valid']['label_neg'] = torch.zeros(split_edge['valid']['edge_neg'].size(0), dtype=torch.float)

    split_edge['test']['label_pos'] = torch.ones(split_edge['test']['edge'].size(0), dtype=torch.float)
    split_edge['test']['label_neg'] = torch.zeros(split_edge['test']['edge_neg'].size(0), dtype=torch.float)
    return split_edge

if __name__ == '__main__':
    load_acm_mat()
