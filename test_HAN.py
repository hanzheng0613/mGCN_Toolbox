import torch
from sklearn.metrics import f1_score
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import * 
import datetime
import errno
import os
import pickle
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from sklearn.cluster import KMeans
from OpenAttMultiGL.layers.hdmi.gcn import GCN
#from OpenAttMultiGL.model.GATNE.utils import *
from OpenAttMultiGL.model.HAN.evaluate import evaluate
import dgl


from dgl.data.utils import _get_dgl_url, download, get_download_dir
import numpy as np

def combine_att(h_list):
    att_act1 = nn.Tanh()
    att_act2 = nn.Softmax(dim=-1)
    h_combine_list = []
    for i, h in enumerate(h_list):
        h = w_list[i](h)
        h = y_list[i](h)
        h_combine_list.append(h)
    score = torch.cat(h_combine_list, -1)
    score = att_act1(score)
    score = att_act2(score)
    score = torch.unsqueeze(score, -1)
    h = torch.stack(h_list, dim=1)
    h = score * h
    h = torch.sum(h, dim=1)
    return h

def embed(seq, adj_list, sparse):
    global w_list
    global y_list
    gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
    w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
    y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])
    h_1_list = []
    for i, adj in enumerate(adj_list):
        h_1 = torch.squeeze(gcn_list[i](seq, adj, sparse))
        h_1_list.append(h_1)
    h = combine_att(h_1_list)
    return h.detach()
def score(logits, labels,num_classes):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    #sim = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")
    nmi = run_kmeans(labels, prediction, num_classes)
    labels = labels.reshape(-1,1)
    sim = run_similarity_search(labels, prediction)
    return nmi, micro_f1, macro_f1,sim


def evaluate(model, g, features, labels, mask, loss_func,num_classes):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    nmi, micro_f1, macro_f1,sim = score(logits[mask], labels[mask],num_classes)
    
    

    return micro_f1, macro_f1,nmi,sim

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]
    sim = []
    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))
    for i in st:
        sim.append(float(i))
    st = ','.join(st)
    
    sim_mean = np.mean(sim)
    #print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))
    return sim
def run_kmeans(y,y_pred, k):
    estimator = KMeans(n_clusters=k,n_init=10)#, n_jobs=16)

    NMI_list = []
    for i in range(5):
        #estimator.fit(x)
        #y_pred = estimator.predict(x)
        s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        e = float("{:.4f}".format(s))
        NMI_list.append(e)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    #print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    return mean

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    global best_macro
    global best_micro
    best_macro = 0
    best_micro = 0
    
    
    sparse = True
    features = torch.FloatTensor(preprocessed_features)
    labels = torch.FloatTensor(c.gcn_labels)
    idx_train = torch.LongTensor(c.train_id)
    idx_val = torch.LongTensor(c.valid_id)
    idx_test = torch.LongTensor(c.test_id)
    
    gcn_adj_list = [normalize_adj(adj) for adj in c.gcn_adj_list]
    adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in gcn_adj_list]
    embeds = embed(features, adj_list, sparse)
    test_embs = embeds[idx_test]

        
    test_lbls = torch.argmax(labels[idx_test], dim=1)
    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())
    #print('test embs:  ', test_embs.shape)
    #print('test lbls:  ', test_lbls.shape)
    
    if dataname == "amazon":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/AMAZON/amazon.pkl', "rb"))
        data["IVI"] = sp.csr_matrix(data["IVI"])
        data["IBI"] = sp.csr_matrix(data["IBI"])
        data["IOI"] = sp.csr_matrix(data["IOI"])
        author_g = dgl.from_scipy(data["IVI"])
        subject_g = dgl.from_scipy(data["IBI"])
        o_g = dgl.from_scipy(data["IOI"])
        gs = [author_g, subject_g, o_g]
    elif dataname == "acm":
        data = sio.loadmat('OpenAttMultiGL/data/HAN/ACM/acm.mat')
        data["PAP"] = sp.csr_matrix(data["PAP"])
        data["PLP"] = sp.csr_matrix(data["PLP"])
        author_g = dgl.from_scipy(data["PAP"])
        subject_g = dgl.from_scipy(data["PLP"])
        gs = [author_g, subject_g]
    elif dataname == "dblp":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/DBLP/dblp.pkl', "rb"))
        data["PAP"] = sp.csr_matrix(data["PAP"])
        data["PPrefP"] = sp.csr_matrix(data["PPrefP"])
        data["PATAP"] = sp.csr_matrix(data["PATAP"])
        author_g = dgl.from_scipy(data["PAP"])
        subject_g = dgl.from_scipy(data["PPrefP"])
        o_g = dgl.from_scipy(data["PATAP"])
        gs = [author_g, subject_g, o_g]
    elif dataname == "imdb":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/IMDB/imdb.pkl', "rb"))
        data["MDM"] = sp.csr_matrix(data["MDM"])
        data["MAM"] = sp.csr_matrix(data["MAM"])
        author_g = dgl.from_scipy(data["MDM"])
        subject_g = dgl.from_scipy(data["MAM"])
        gs = [author_g, subject_g]
    
    num_classes = c.gcn_labels.shape[1]
    c.gcn_labels = torch.from_numpy(data["label"]).long()
    c.gcn_labels = c.gcn_labels.nonzero()[:, 1]
    c.features = c.features.toarray()
    c.features = torch.from_numpy(data["feature"]).float()
    num_nodes = author_g.num_nodes()
    train_mask = get_binary_mask(num_nodes, c.train_id)
    val_mask = get_binary_mask(num_nodes, c.valid_id)
    test_mask = get_binary_mask(num_nodes, c.test_id)
    
    #t = dataset(args["dataset"])
    #print(type(t.edge_index))
    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    c.features = c.features.to(args["device"])
    c.gcn_labels = c.gcn_labels.to(args["device"])
    
    
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])
    
    
    #print(args["hetero"])
    if args["hetero"]:
        from OpenAttMultiGL.HAN.model_hetero import HAN
        
        model = HAN(
            meta_paths=[["pa", "ap"], ["pf", "fp"]],
            in_size=c.features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=t.HAN_num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        gs = gs.to(args["device"])
        
    else:
        from OpenAttMultiGL.model.HAN.model import HAN

        model = HAN(
            num_meta_paths=len(gs),
            in_size=c.features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        gs = [graph.to(args["device"]) for graph in gs]

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    micro_list = []
    macro_list = []
    nmi_list = []
    sim_list = []
    #for epoch in range(1):
    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(gs, c.features)
        loss = loss_fcn(logits[train_mask], c.gcn_labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_micro_f1, test_macro_f1,nmi,sim = evaluate(
            model, gs, c.features, c.gcn_labels, test_mask, loss_fcn,num_classes
        )
        
        #sim = run_similarity_search(test_embs, test_lbls)
        sim = sim[0]
        if test_macro_f1 > best_macro and test_micro_f1> best_micro:
            best_macro = test_macro_f1
            best_micro = test_micro_f1
            print('Epoch:', epoch)
        #print("Best Validation:", t1)
            print("Macro_F1:", test_macro_f1)
            print("Micro_F1:", test_micro_f1)
            print("NMI: ", nmi)
            print("SIM: ", sim)
        macro_list.append(test_macro_f1)
        micro_list.append(test_micro_f1)
        nmi_list.append(nmi)
        sim_list.append(sim)
    
    print("Final score: ")
    print('Micro: {:.4f} ({:.4f})'.format(np.mean(micro_list),np.std(micro_list)))
    print('Macro: {:.4f} ({:.4f})'.format(np.mean(macro_list),np.std(macro_list)))
    print('Sim: {:.4f} ({:.4f})'.format(np.mean(sim_list),np.std(sim_list)))
    print('NMI: {:.4f} ({:.4f})'.format(np.mean(nmi_list),np.std(nmi_list)))


if __name__ == "__main__":
    import argparse
    dataname = "imdb"
    c = dataset(dataname)
    preprocessed_features = preprocess_features(c.features)
    ft_size = preprocessed_features[0].shape[1] 
    hid_units = 128
    n_networks = len(c.adj_list)
    #from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)