#!usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import time
import random
# import tensorflow as tf
import numpy as np
import scipy
import scipy.sparse as sp
from sklearn.cluster import KMeans
from OpenAttMultiGL.model.MvAGC.metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from time import *
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter('error',ComplexWarning)
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import * 




def FGC_cora_modified(X, av, gnd, a, k, ind):
        # Store some variables
    gama=-1
    nada = [1, 1]
    X_hat_list=[]
    X_hat_anchor_list=[]
    A_hat_list=[]
    final = []
    for i in range(2):
        A=av[i]
        N = X.shape[0]
        # print("N = {}".format(N))
        Im = np.eye(len(ind))
        In = np.eye(N)
        if sp.issparse(X):
            X = X.todense()

        # Normalize A
        A = A + In
        D = np.sum(A, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        A = D.dot(A).dot(D)

        # Get filter G
        Ls = In - A
        G = In - 0.2 * Ls #0.5 #0.8
        G_ = In
        X_hat = X
        for i in range(k):
            # G_ = G_.dot(G)
            X_hat = G.dot(X_hat)
        X_hat_list.append(X_hat)
        A_hat = A[ind]  # (m,n)
        A_hat_list.append(A_hat)
        X_hat_anchor_list.append(X_hat[ind])
    begin_time = time()
    # Set the order of filter
    for t in range(20):
        tmp1=0
        tmp2=0
        for i in range(2):
            tmp1 =tmp1+nada[i]*(X_hat_anchor_list[i].dot(X_hat_anchor_list[i].T) + a * Im)
        for i in range(2):
            tmp2 = tmp2+nada[i]*(X_hat_anchor_list[i].dot(X_hat_list[i].T) + a * A_hat_list[i])
        S = np.linalg.inv(tmp1).dot(tmp2)
        for i in range(2):
            nada[i] = (-((np.linalg.norm(X_hat_list[i].T - (X_hat_anchor_list[i].T).dot(S))) ** 2 + a * (np.linalg.norm(S - A_hat_list[i])) ** 2) / (gama)) ** (1 / (gama - 1))
            #print("nada value")
            #print(nada[i])
        # res=0
        # for j in range(2):
        #     res = res + nada[j] * ((np.linalg.norm(X_hat_list[i].T - (X_hat_anchor_list[i].T).dot(S))) ** 2 + a * (np.linalg.norm(S - A_hat_list[i])) ** 2) + (nada[j]) ** (gama)
        # final.append(res)
        # print(res)
    # sio.savemat("a.mat", {'res': final})
    return S, begin_time


def main(X, av, gnd, m, a, k, ind):

    N = X.shape[0]
    begin_time_filter = time()
    types = len(np.unique(gnd))
    S, begin_time = FGC_cora_modified(X, av, gnd, a, k, ind)
    D = np.sum(S, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)  # (m,m)

    S_hat = D.dot(S)  # (m,n)

    S_hat_tmp = S_hat.dot(S_hat.T)  # (m,m)
    S_hat_tmp[np.isinf(S_hat_tmp)] = 0
    S_hat_tmp[np.isnan(S_hat_tmp)] = 0
    # sigma, E = scipy.linalg.eig(S_hat_tmp)
    E, sigma, v = sp.linalg.svds(S_hat_tmp, k=types, which='LM')
    sigma = sigma.T
    sigma = np.power(sigma, -0.5)
    sigma[np.isinf(sigma)] = 0
    sigma[np.isnan(sigma)] = 0
    sigma = np.diagflat(sigma)
    C_hat = (sigma.dot(E.T)).dot(S_hat)
    C_hat[np.isinf(C_hat)] = 0
    C_hat[np.isnan(C_hat)] = 0
    C_hat = C_hat.astype(float)
    #print("print shape of C_hat: ", C_hat.shape)
    kmeans = KMeans(n_clusters=types, random_state=37).fit(C_hat.T)

    predict_labels = kmeans.predict(C_hat.T)

    cm = clustering_metrics(gnd, predict_labels)
    f1_micro, f1_macro, nmi,sim = cm.evaluationClusterModelFromLabel(m,a,k)
    end_time = time()
    tot_time = end_time - begin_time
    tot_time_filter = end_time - begin_time_filter
    return f1_micro, f1_macro, nmi,sim


def lower_bound(p, rd):
    l = 0
    r = len(p) - 1
    while(l < r):
        # print("rd = {}, l = {}, r= {}".format(rd, l, r))
        mid = (l + r) // 2
        if(p[mid] > rd):
            r = mid
        else:
            l = mid + 1
    # print("rd = {}, l = {}, r= {}".format(rd, l, r))
    return l


def node_sampling(A, m, alpha):
    D = np.sum(A[0], axis=1).flatten()+np.sum(A[1], axis=1).flatten()

    if(len(np.shape(D)) > 1):
        D = D.A[0]
        print(1)

    D = D**alpha
    D=D/10000
    #print(D)
    tot = np.sum(D)
    #print(tot)
    p = D / tot
    #print(p)
    for i in range(len(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    #print(p)
    ind = []
    vis = [0] * len(D)
    while(m):
        while(1):
            rd = np.random.rand()
            pos = lower_bound(p, rd)
            if(vis[pos] == 1):
                continue
            else:
                vis[pos] = 1
                ind.append(pos)
                m = m - 1
                break
    return ind


def func(X, A, gnd):
    m_init_list = [60] #anchor numbers
    a_list = [5,15] #second term
    k_init_list = [2] #juanjijieshu
    f_alpha_init_list = [4] #important node
    k_list = []
    aa_list = []
    i_list = []
    ac_list = []
    nm_list = []
    f1_list = []
    adj_list=[]
    tm_list = []
    tm_list_filter = []
    f_alpha_list = []

    N = X.shape[0]
    tot_test = 1
    ac_max = 0.0
    xia = 0
    tot = 0

    # print(node_sampling(A, 20))
    for k in k_init_list:
        for i in m_init_list:
            # print("now k = {}, now m = {}".format(k, i))
            for alpha in f_alpha_init_list:
                ind = node_sampling(A, i, alpha)
                ac_mean = 0
                nm_mean = 0
                f1_mean = 0
                adj_mean=0
                tm_mean = 0
                for a in a_list:
                    # continue
                    f1_micro, f1_macro, nmi,sim = main(
                        X, A, gnd, i, a, k, ind)
                    
                

                if(ac_mean > ac_max):
                    xia = tot
                    ac_max = ac_mean

                tot += 1

    
    return f1_micro, f1_macro, nmi,sim


if __name__ == '__main__':
    global best_macro
    global best_micro
    global best_hits
    best_macro = 0
    best_micro = 0
    dataname = 'imdb'#'ACM3025'#'AMAZON3025'
    s = dataset(dataname)
    num_classes = s.gcn_labels.shape[1]
    #print("numclass ",num_classes)
    s.gcn_labels = s.gcn_labels.T
    
    s.gcn_labels = np.argmax(s.gcn_labels, axis=0)
    #data = pkl.load(open('mGCN_Toolbox/data/HAN/AMAZON/amazon.pkl', "rb"))
    #print(type(data["feature"]))
    #s.features = np.array(s.features)
    #print("T:",s.features.shape)
    #print(type(s.features))
    if dataname == "amazon":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/AMAZON/amazon.pkl', "rb"))
        A = data["IVI"]
        B = data["IBI"]
        C = data["IOI"]
        av=[]
        av.append(A)
        av.append(B)
        av.append(C)
    elif dataname == "acm":
        data = sio.loadmat('OpenAttMultiGL/data/HAN/ACM/acm.mat')
        A = data['PAP']
        B = data['PLP']
        av=[]
        av.append(A)
        av.append(B)
    elif dataname == "dblp":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/DBLP/dblp.pkl', "rb"))
        A = data["PAP"]
        B = data["PPrefP"]
        C = data["PATAP"]
            
           
        av=[]
        av.append(A)
        av.append(B)
        av.append(C)
    elif dataname == "imdb":
        data = pkl.load(open('OpenAttMultiGL/data/HAN/IMDB/imdb.pkl', "rb"))
        A = data["MDM"]
        B = data["MAM"]
            
        
        av=[]
        av.append(A)
        av.append(B)
    # number of epoch
    tt = 100
    #print(tt)
    micro_list = []
    macro_list = []
    nmi_list = []
    sim_list = []
    for i in range(tt):
        #print('features', type(s.features))
        #print('av', type(av))
        #print('label',type(s.gcn_labels))
        f1_micro, f1_macro, nmi,sim = func(s.features.toarray(), av, s.gcn_labels)
        if f1_macro > best_macro and f1_micro> best_micro:
            best_macro = f1_macro
            best_micro = f1_micro
            print('Epoch:', i)
        #print("Best Validation:", t1)
            print("Macro_F1:", f1_macro)
            print("Micro_F1:", f1_micro)
            print("NMI: ", nmi)
            print("SIM: ", sim)
        micro_list.append(f1_micro)
        macro_list.append(f1_macro)
        nmi_list.append(nmi)
        sim_list.append(sim)
    print("Mean: f1_micro = {}, f1_macro = {},nmi = {}, sim = {}\n".format(
                np.mean(micro_list),np.mean(macro_list),np.mean(nmi_list),np.mean(sim_list)))
    print("Std: f1_micro = {}, f1_macro = {},nmi = {}, sim = {}\n".format(
                np.std(micro_list),np.std(macro_list),np.std(nmi_list),np.std(sim_list)))