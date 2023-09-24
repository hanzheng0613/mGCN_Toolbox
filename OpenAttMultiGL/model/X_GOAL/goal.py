'''
     The PyTorch implementation of X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning\
     
     https://github.com/baoyujing/X-GOAL/tree/main    
'''

import os
from tqdm import tqdm


import torch

from .model import Model
from .encoder import Encoder

from OpenAttMultiGL.model.X_GOAL.evaluate import evaluate
from sklearn import metrics

import numpy as np
import argparse
from OpenAttMultiGL.model.mGCN.mGCN_node import*
from OpenAttMultiGL.utils.process import * 
from OpenAttMultiGL.layers.hdmi.gcn import GCN
import torch.nn as nn
import torch.optim as optim
import torch
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import split_node_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import pairwise

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

def embed(seq, adj_list, sparse,n_networks,ft_size):
    global w_list
    global y_list

    hid_units = 128
    gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
    w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
    y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])
    h_1_list = []
    for i, adj in enumerate(adj_list):
        h_1 = torch.squeeze(gcn_list[i](seq, adj, sparse))
        h_1_list.append(h_1)
    h = combine_att(h_1_list)
    return h.detach()

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


class GOAL(Model):
    def __init__(self, args):
        super().__init__(args)
        self.adj = self.adj_list[self.args.layer].to(self.args.device)
        self.encoder = Encoder(self.args.ft_size, self.args.hid_units).to(self.args.device)
        self.embeds = self.get_embeddings()
        

    def get_embeddings(self):
        self.encoder.eval()
        embeds = self.encoder(self.features, self.adj)
        embeds = embeds.detach()
        return embeds

    def train(self):
        print("Started training on {}-{} layer with {}...".format(self.args.dataset, self.args.layer, self.args.model))
        if self.args.is_warmup:
            self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.warmup_lr)
            self.warmup()
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        if self.args.pretrained_model_path:
            path = self.args.pretrained_model_path
        else:
            path = os.path.join(self.args.save_root, 'warmup_{}_{}_{}.pkl'.format(
                self.args.dataset, self.args.model, self.args.layer))
        #self.encoder.load_state_dict(torch.load(path))
        self._train_full_loss()

    def evaluate(self, path=""):
        if path:
            print("Evaluating based on {}".format(path))
            self.encoder.load_state_dict(torch.load(path))
        embs = self.get_embeddings()
        macro_f1s, micro_f1s, nmi, sim = evaluate(embs, self.idx_train, self.idx_val, self.idx_test, self.labels)
        return macro_f1s, micro_f1s, nmi, sim
    def _train_full_loss(self):
        print("Full training loss...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()

            feature_pos, adj_pos = self.transform_pos(self.features, self.adj)
            feature_neg, adj_neg = self.transform_neg(self.features, self.adj)

            h_org = self.encoder(self.features, self.adj)
            h_pos = self.encoder(feature_pos, adj_pos)
            h_neg = self.encoder(feature_neg, adj_neg)

            loss_nce = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
            if n_epoch % self.args.cluster_step == 0:
                centroids, target = self.run_kmeans(x=h_org, k=self.args.k)
            loss_cluster = self.loss_kmeans(x=h_org, centroids=centroids, tau=self.args.tau, target=target)
            loss = loss_nce + self.args.w_cluster*loss_cluster
            if n_epoch % 10 == 0:
                print(loss, loss_nce, loss_cluster)

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                torch.save(self.encoder.state_dict(), os.path.join(self.args.save_root, '{}_{}_{}.pkl'.format(
                    self.args.dataset, self.args.model, self.args.layer)))
                break

            loss.backward()
            self.opt.step()

    def warmup(self):
        print("Warming up...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()

            feature_pos, adj_pos = self.transform_pos(self.features, self.adj)
            feature_neg, adj_neg = self.transform_neg(self.features, self.adj)

            h_org = self.encoder(self.features, self.adj)
            h_pos = self.encoder(feature_pos, adj_pos)
            h_neg = self.encoder(feature_neg, adj_neg)

            loss = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
            if n_epoch % 100 == 0:
                print("L_n: {:.6f}".format(loss.detach().cpu().numpy()))

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                torch.save(self.encoder.state_dict(), os.path.join(self.args.save_root, 'warmup_{}_{}_{}.pkl'.format(
                    self.args.dataset, self.args.model, self.args.layer)))
                break

            loss.backward()
            self.opt.step()
