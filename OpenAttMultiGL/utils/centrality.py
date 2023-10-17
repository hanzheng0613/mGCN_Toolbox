"Original python file comes from "
"https://www.math.ucla.edu/~mason/research/Flora_BEE_Submitted-032315.pdf"

import numpy as np
import csv
from random import*
from numpy import*
from numpy import linalg as la
#from itertools import izip

from OpenAttMultiGL.utils.dataset import dataset

def centrality_implementation(dataname):
    t = dataset(dataname)
    #K=t.features.shape[0] # number of nodes
    K=100
    L=len(t.truefeatures_list) # number of layers
    N=K*L # number of (node,layer ) pairs

    pvalues=L*[0.25] # edge probability vector
    edgemin=1 # minimum intra layer edge weight
    edgemax=38000 # maximum intra layer edge weight
    coupling_strength =400 # strength of layer coupling
    simulation_number =1
    
                       

    global SAM
    SAM=np.zeros(shape=(N,N))
    for i in range(len(t.gcn_adj_list)):
        t.gcn_adj_list[i] = t.gcn_adj_list[i].toarray()
    for l in range(L):
            #intraAM=np.random.binomial(1,pvalues[l],K**2 ) # Generate edges (repeat a Bernoulli process with success ratep [l] K**2 times )
    
            #intraAM=intraAM.reshape((K,K))
            #intraAM [range(K),range(K)]=0 # make diagonal zero (remove self−loops)
            #intraAM=np.triu(intraAM) # make sub−diagonal entries all zero
        intraAM = np.zeros(shape=(N,N))
    
        for j in range(K):
            for k in range(K):
                intraAM[l*K+j][l*K+k] = t.gcn_adj_list[l][j][k]
                    
        for i in range (K):
            for j in range (K):
                if intraAM [i,j]==1:
                    intraAM [i,j]= uniform (edgemin,edgemax) ## Add random edge weights to intralayer edges
        for i in range (K) :
            for j in range (K-i-1) :
                intraAM [i+j+1][i]=intraAM[i][i+j+1] # reflect intraAM in its diagonal
        for i in range (K) :
            for j in range (K) :
                SAM[i+K*l][j+K*l]=intraAM[i][j] # incorporate intraAM in to the corresponding diagonal block of SAM
    SAM=np.asmatrix(SAM)

    ## Add couplings
    for node in range (K) :
        for layer1 in range(L) :
            for layer2 in range(L) :
                if layer2 != layer1:
                    SAM[node+K*layer1,node+K*layer2]= coupling_strength

## Transition matrix for the multiplex network
    S=np.sum(SAM,axis=1) # Note that the multiplex network is undirectedso the supra−adjacency matrix is symmetric
    d=np.zeros(N)
    for i in range(N) :
        d[i]=1/S[i] # Note that S[i] is non−zero because of couplings
    D=np.diag([d[i] for i in range(N)])
    T=np.dot(SAM,D) # Note that T is a left stochastic matrix,i.e.all entries non−negative and each column sums to 1

## Get a monoplex network by aggregating the multiplex network across layers
#i.e. add all the intralayer matrices,which are weighted by layer couplings
    aggAM=np.zeros((K,K))
    for l in range(L):
        for i in range(K):
            for j in range(K) :
                aggAM[i,j]=aggAM[i,j]+SAM[i+l*K,j+l*K]

## Calculate the transition matrix for the aggregated network
    aggS=np.sum(aggAM, axis=1)
    agg_isolated=set(np.where(aggS==0)[0]) # the index set of isolated nodes in the aggregated network
    aggS[aggS==0]=1 # This is to avoid division by zero later.Columns of aggT that correspond to isolated point sare still all zero
    aggd=np.zeros(K)
    for i in range(K):
        aggd[i]=1/aggS[i]
    aggD=np.diag([aggd[i] for i in range (K)])
    aggT=np.dot(aggAM,aggD) # Similar to T, aggT is a left stochastic matrix


    #print('A random multiplex network has been generated.')
    return SAM

def RWOC(SAM,dataname):
## RWOC
    t = dataset(dataname)
    #K=t.features.shape[0] # number of nodes
    K=100
    L=len(t.truefeatures_list) # number of layers
    N=K*L # number of (node,layer ) pairs
    weight=sum(SAM)
    rankRWOC=np.zeros(K)
    for i in range (K):
        k=0
        for l in range (L):
            k=k+sum(SAM[i+l*K, :])
        rankRWOC[i]=k/weight
    print('RWOC done.')
    return rankRWOC

def simple_RWOC(SAM,dataname):
## Simple RWOC
    t = dataset(dataname)
    K=t.features.shape[0] # number of nodes
    L=len(t.truefeatures_list) # number of layers
    N=K*L # number of (node,layer ) pairs
    simrankRWOC=np.zeros(K)
    for l in range(L):
        intralayer=np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                intralayer[i][j]=SAM[i+l*K,j+l*K] # intralayer is the adjacency matrix for layer l
        weight=sum(intralayer)
        for h in range(K):
            k=sum(intralayer[h,:])
            simrankRWOC[h]=simrankRWOC[h]+k/weight
    simrankRWOC=simrankRWOC/L
    print('simRWOC done.')

def aggregate_RWOC(aggAM,dataname):
    ## Aggregate RWOC
    t = dataset(dataname)
    K=t.features.shape[0] # number of nodes
    L=len(t.truefeatures_list) # number of layers
    N=K*L # number of (node,layer ) pairs
    weight=sum(aggAM)
    aggrankRWOC=np.zeros(K)
    for i in range(K):
        aggrankRWOC[i]=sum(aggAM[i,:])/weight
    print('aggRWOC done')

def page_rank(SAM,D,dataname):
## PageRank
    t = dataset(dataname)
    K=t.features.shape[0] # number of nodes
    L=len(t.truefeatures_list) # number of layers
    N=K*L # number of (node,layer ) pairs
    T=np.dot(SAM,D) # Refresh matrix T
    r =0.85 # (1−r ) is the teleportation rate
    R=r*T+(1-r)/N*np.ones((N,N))
    RLeigenvector=np.asmatrix(np.ones(N))
    RLeigenvector=RLeigenvector.transpose() # Now RLeigenvectoris a N*1 column vector
    while la.norm(RLeigenvector-np.dot(R, RLeigenvector))>10**-8:
        RLeigenvector=np.dot(R, RLeigenvector)/la.norm(np.dot(R,RLeigenvector))
    NRLeigenvector=RLeigenvector/np.sum(RLeigenvector)
    NRLeigenvector=NRLeigenvector.reshape((L,K))
    rankPageRank=np.squeeze(np.asarray(np.sum(NRLeigenvector,axis=0)))
    print('PageRank done.')