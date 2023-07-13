import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
import datetime
import errno
import os
import pickle
import random
from pprint import pprint


import dgl

from dgl.data.utils import _get_dgl_url, download, get_download_dir

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils import to_undirected

import dgl


def load_acm_mat(sc=3):
    data = sio.loadmat('mGCN_Toolbox/data/acm/acm.mat')
    label = data['label']
    

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc

    sequence_adj1 = data["PLP"]
    sequence_adj2 = data["PAP"]
    
    sequence_adj = [sequence_adj1,sequence_adj2]
    
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]
    
    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)
    
    truefeatures_list = []
    for _ in range(len(adj_list)):
        truefeatures_list.append(truefeatures)
        
    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures_list, truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list,sequence_adj



def load_dblp(sc=3):
    data = pkl.load(open("mGCN_Toolbox/data/dblp/dblp.pkl", "rb"))
    label = data['label']

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc
    
    sequence_adj1 = data["PAP"]
    sequence_adj2 = data["PPrefP"]
    sequence_adj3 = data["PATAP"]
    
    sequence_adj = [sequence_adj1,sequence_adj2,sequence_adj3]

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)
    
    truefeatures_list = []
    for _ in range(len(adj_list)):
        truefeatures_list.append(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures_list, truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list,sequence_adj


def load_imdb(sc=3):
    data = pkl.load(open("mGCN_Toolbox/data/imdb/imdb.pkl", "rb"))
    label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc
    
    sequence_adj1 = data["MDM"]
    sequence_adj2 = data["MAM"]
    
    sequence_adj = [sequence_adj1,sequence_adj2]

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]
    
    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)
    
    truefeatures_list = []
    for _ in range(len(adj_list)):
        truefeatures_list.append(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures_list, truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list,sequence_adj


def load_amazon(sc=3):
    data = pkl.load(open("mGCN_Toolbox/data/amazon/amazon.pkl", "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc
    
    sequence_adj1 = data["IVI"]
    sequence_adj2 = data["IBI"]
    sequence_adj3 = data["IOI"]
    
    sequence_adj = [sequence_adj1,sequence_adj2,sequence_adj3]

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)
    
    truefeatures_list = []
    for _ in range(len(adj_list)):
        truefeatures_list.append(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return truefeatures_list, truefeatures, edge_index, len(edge_index), label, idx_train, idx_val, idx_test, adj_list,sequence_adj

def load_txt_training_data(f_name):
    #print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_txt_testing_data(f_name):
    #print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type

def load_txt_feature_data(f_name):
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        log_dir = log_dir + "_sampling"

    mkdir_p(log_dir)
    return log_dir

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix

# The configuration below is from the paper.
default_configure = {
    "lr": 0.005,  # Learning rate
    "num_heads": [8],  # Number of attention heads for node-level attention
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "num_epochs": 10,
    "patience": 100,
}

sampling_configure = {"batch_size": 20}

def setup(args):
    args.update(default_configure)
    set_random_seed(args["seed"])
    # args["dataset"] = "ACMRaw" if args["hetero"] else "ACM"
    args["dataset"] = "DBLP"
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_IMDB(remove_self_loop):
    data = pkl.load(open('mGCN_Toolbox/data/HAN/IMDB/imdb.pkl', "rb"))
    # label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*3
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*3

    
    labels, features = (
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["feature"]).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data["label"].shape[0]
        data["MDM"] = sp.csr_matrix(data["MDM"] - np.eye(num_nodes))
        data["MAM"] = sp.csr_matrix(data["MAM"] - np.eye(num_nodes))
    else:
        data["MDM"] = sp.csr_matrix(data["MDM"])
        data["MAM"] = sp.csr_matrix(data["MAM"])
    
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    
    adj_list = [adj1, adj2]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]
    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data["MDM"])
    subject_g = dgl.from_scipy(data["MAM"])
    gs = [author_g, subject_g]

    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * 0.3)
    valid_sample = int(len(labels) * 0.1)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    train_idx = torch.LongTensor(training_ind)
    val_idx = torch.LongTensor(valid_id)
    test_idx = torch.LongTensor(test_id)
    # train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
    # val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
    # test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)

    num_nodes = author_g.num_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("dataset loaded")
    pprint(
        {
            "dataset": "IMDB",
            "train": train_mask.sum().item() / num_nodes,
            "val": val_mask.sum().item() / num_nodes,
            "test": test_mask.sum().item() / num_nodes,
        }
    )

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        edge_index
    )

def load_AMAZON(remove_self_loop):
    data = pkl.load(open('mGCN_Toolbox/data/HAN/AMAZON/amazon.pkl', "rb"))
    # label = data['label']
    # adj1 = data["IVI"]
    # adj2 = data["IBI"]
    # adj3 = data["IOI"]
    # adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    # adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc
    random.seed(0)

    labels, features = (
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["feature"]).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data["label"].shape[0]
        data["IVI"] = sp.csr_matrix(data["IVI"] - np.eye(num_nodes))
        data["IBI"] = sp.csr_matrix(data["IBI"] - np.eye(num_nodes))
        data["IOI"] = sp.csr_matrix(data["IOI"] - np.eye(num_nodes))
    else:
        data["IVI"] = sp.csr_matrix(data["IVI"])
        data["IBI"] = sp.csr_matrix(data["IBI"])
        data["IOI"] = sp.csr_matrix(data["IOI"])
    
    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*3
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*3
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*3
    
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    
    adj_list = [adj1, adj2, adj3]

    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data["IVI"])
    subject_g = dgl.from_scipy(data["IBI"])
    o_g = dgl.from_scipy(data["IOI"])
    gs = [author_g, subject_g, o_g]
    training_percent = 0.2
    # train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
    # val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
    # test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)
    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * training_percent)
    valid_sample = int(len(labels) * (training_percent / 2))
    np.random.shuffle(ind)
    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * 0.3)
    valid_sample = int(len(labels) * 0.1)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    train_idx = torch.LongTensor(training_ind)
    val_idx = torch.LongTensor(valid_id)
    test_idx = torch.LongTensor(test_id)
    # train_idx = torch.LongTensor(ind[:training_sample])
    # val_idx = torch.LongTensor(ind[training_sample:training_sample + valid_sample])
    # test_idx = torch.LongTensor(ind[training_sample + valid_sample:])

    num_nodes = author_g.num_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("dataset loaded")
    pprint(
        {
            "dataset": "AMAZON",
            "train": train_mask.sum().item() / num_nodes,
            "val": val_mask.sum().item() / num_nodes,
            "test": test_mask.sum().item() / num_nodes,
        }
    )

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        edge_index
    )

def load_ACM(remove_self_loop):
    
    data = sio.loadmat('mGCN_Toolbox/data/HAN/ACM/acm.mat')
    labels, features = (
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["feature"]).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    
    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*3
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*3
    
    if remove_self_loop:
        num_nodes = data["label"].shape[0]
        data["PAP"] = sp.csr_matrix(data["PAP"] - np.eye(num_nodes))
        data["PLP"] = sp.csr_matrix(data["PLP"] - np.eye(num_nodes))
    else:
        data["PAP"] = sp.csr_matrix(data["PAP"])
        data["PLP"] = sp.csr_matrix(data["PLP"])
    
    
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]
    
    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1)]
    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PLP"])
    gs = [author_g, subject_g]
    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * 0.3)
    valid_sample = int(len(labels) * 0.1)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    train_idx = torch.LongTensor(training_ind)
    val_idx = torch.LongTensor(valid_id)
    test_idx = torch.LongTensor(test_id)
    # train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
    # val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
    # test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)

    num_nodes = author_g.num_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("dataset loaded")
    pprint(
        {
            "dataset": "ACM",
            "train": train_mask.sum().item() / num_nodes,
            "val": val_mask.sum().item() / num_nodes,
            "test": test_mask.sum().item() / num_nodes,
        }
    )

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        edge_index
    )

def load_DBLP(remove_self_loop):
    
    #data = sio.loadmat('mGCN_Toolbox/data/HAN/DBLP/dblp.pkl')
    data = pkl.load(open('mGCN_Toolbox/data/HAN/DBLP/dblp.pkl', "rb"))
    labels, features = (
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["feature"]).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data["label"].shape[0]
    
    else:
        data["PAP"] = sp.csr_matrix(data["PAP"])
        data["PPrefP"] = sp.csr_matrix(data["PPrefP"])
        data["PATAP"] = sp.csr_matrix(data["PATAP"])
    
    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*3
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*3
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*3
    
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]
    #print("adj1:",adj1)
    edge_index = [from_scipy_sparse_matrix(adj1)[0].transpose(0, 1), from_scipy_sparse_matrix(adj2)[0].transpose(0, 1), from_scipy_sparse_matrix(adj3)[0].transpose(0, 1)]
    
    print("type: ", type(edge_index))
    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PPrefP"])
    o_g = dgl.from_scipy(data["PATAP"])
    gs = [author_g, subject_g, o_g]
    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * 0.3)
    valid_sample = int(len(labels) * 0.1)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    train_idx = torch.LongTensor(training_ind)
    val_idx = torch.LongTensor(valid_id)
    test_idx = torch.LongTensor(test_id)
    # train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
    # val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
    # test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)

    num_nodes = author_g.num_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("dataset loaded")
    pprint(
        {
            "dataset": "DBLP",
            "train": train_mask.sum().item() / num_nodes,
            "val": val_mask.sum().item() / num_nodes,
            "test": test_mask.sum().item() / num_nodes,
        }
    )

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        edge_index
    )

def load_ACM_raw(remove_self_loop):
    assert not remove_self_loop
    url = "mGCN_Toolbox/data/HAN/ACM_RAW.mat"
    data_path = get_download_dir() + "/ACM_RAW.mat"
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data["PvsL"]  # paper-field?
    p_vs_a = data["PvsA"]  # paper-author
    p_vs_t = data["PvsT"]  # paper-term, bag of words
    p_vs_c = data["PvsC"]  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): p_vs_a.nonzero(),
            ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
            ("paper", "pf", "field"): p_vs_l.nonzero(),
            ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
        }
    )

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    ind = np.arange(0, len(labels), 1)
    training_sample = int(len(labels) * 0.3)
    valid_sample = int(len(labels) * 0.1)
    np.random.shuffle(ind)
    training_ind = ind[:training_sample]
    valid_id = ind[training_sample:training_sample + valid_sample]
    test_id = ind[training_sample + valid_sample:]
    train_idx = torch.LongTensor(training_ind)
    val_idx = torch.LongTensor(valid_id)
    test_idx = torch.LongTensor(test_id)
    # train_idx = np.where(float_mask <= 0.2)[0]
    # val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    # test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.num_nodes("paper")
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )
class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))



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



def sequence_normalize_adj(adj):
    """Sequentially normalize adjacency matrix in a symmetrical way."""
    rowsum = np.array(adj.sum(1))
    zero_rows = rowsum == 0
    adj[zero_rows, zero_rows] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

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

def split_link_data(data, test_view, neg_k, multi=False, R=0):

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
        if R > 0:
            candidate_edges = get_masked(data.x, data.edge_index[test_view], R, split_edge[test_view]['test']['edge'].numpy())
            print("The number of overlapping candidate edges:")
            print(intersect2D(candidate_edges.numpy(), to_undirected(split_edge[test_view]['test']['edge'].transpose(0,1)).transpose(0,1).numpy()))
            print(len(candidate_edges))
            print(len(split_edge[test_view]['test']['edge']))
            return data, split_edge, candidate_edges
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