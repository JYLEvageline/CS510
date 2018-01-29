# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:58:00 2017

@author: evaljy
"""

import math
import operator
import random
import gzip
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import log


def vocab(line):
    vocabulary = {}
    string = line.split()
    for word in string:
        if word in vocabulary:
            vocabulary[word] = vocabulary[word]+1
        else:
            vocabulary[word] = 1
    return vocabulary

def vocab_norm(vocab):
    norm = sum(vocab.values())
    for key in vocab.keys():
        vocab[key] = float(vocab[key])/norm

def _rand_mat(size):
    ret = []
    for i in range(size):
        ret.append(random.random())
    norm = sum(ret)
    for i in range(size):
        ret[i] /= norm
    return ret

def init(docs_prop,K,lam,seed):
    #generate p(w|theta_j)
    docs = docs_prop.docs
    wct = docs_prop.wct
    vocab = docs_prop.vocab
    doc_num = len(docs)
    np.random.seed(seed)
    pw_theta = np.random.dirichlet(np.ones(len(vocab)),size=K)
    #generate pi_{d,j}
    np.random.seed(seed)
    pi_dj = np.random.dirichlet(np.ones(K),size=doc_num)
    #generate p(w|theta_B)
    background = np.sum(wct,axis=0)/np.sum(wct)
    pw_b = np.array([background,]*K)
    return pw_theta,pi_dj,pw_b

def emstep(pw_theta,pi_dj,pw_b,docs_prop,K,lam):
    #zdwb&zdwj doc->word->topic
    wct = docs_prop.wct
    wct_sparse = docs_prop.wct_sparse
    #del(docs_prop)
    change = 1
    changelist = []
    loglist = []
    iteration = 0
    while abs(change) > 0.0001:
        iteration +=1     
        divide = np.dot(pi_dj,(1-lam)*pw_theta+lam*pw_b)
        c = np.divide(wct,divide)

        nd_k = np.dot(c,pw_theta.T)*(1-lam)*pi_dj
        nw_k = np.dot(c.T,(1-lam)*pi_dj)*pw_theta.T
        
        ndk_row_sums = nd_k.sum(axis=1)
        pi_dj = nd_k / ndk_row_sums[:, np.newaxis]
        del(nd_k)
        del(ndk_row_sums)
        
        nwk_col_sums = nw_k.sum(axis=0)
        pw_theta = (nw_k / nwk_col_sums[np.newaxis,:]).T
        print( 'the %sth iteration'%iteration)
        
        log_c = 0
        for pair in wct_sparse:
            [i,j] = pair
            log_c += float(wct[i,j])*log(divide[i,j])
        log_c*=4
        print( 'likelihood %s'%log_c)
        if iteration >1:
            change = (log_p-log_c)/log_p
            changelist.append(change)
            loglist.append(log_c)
        print( 'change %s'%change)
        log_p = log_c
        del(divide)
        del(c)
    return(iteration, loglist,changelist,pw_theta)

def pl_log(loglist,changelist,title):
    fig, ax1 = plt.subplots()
    t = np.arange(0, len(loglist),1)
    s1 = loglist
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('iteration')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('log likelihood', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    s2 = changelist
    ax2.plot(t, s2, 'g-')
    ax2.set_ylabel('relative difference', color='g')
    ax2.tick_params('y', colors='g')
    
    fig.tight_layout()
    
    plt.title(title)
    plt.show()

def get_topicwords(n,K,pw_theta,vcb_idx):
	whole_wl = {}
	for i in range(K):
	    print('Topic %s'%i)
	    print( 20*'-')
	    wl = []
	    idx_list = pw_theta[i,].argsort()[-n:][::-1]
	    for idx in idx_list:
	        for k,v in vcb_idx.items():
	            if v == idx:
	                wl.append(k)
	    print(wl)
	    whole_wl[i] = wl
	return whole_wl

class Doc:
    def __init__(self,docs):
        self.docs = docs
        self.vocab()
        self.wct()
    def vocab(self):
        docs = self.docs
        vocab = list([w for doc in docs for w in doc])
        vocab = list(set(vocab))
        vcb_idx = dict(zip(vocab,range(len(vocab))))
        self.vocab = vocab
        self.vcb_idx = vcb_idx
    def wct(self):
        docs = self.docs
        vocab = self.vocab
        vcb_idx = self.vcb_idx
        wct = np.zeros((len(docs),len(vocab)))
        wct_sparse = []
        for i in range(len(docs)):
    	    for w in docs[i]:
    	        widx = vcb_idx[w]
    	        wct[i,widx] += 1
    	        wct_sparse.append([i,widx])
        self.wct = wct
        self.wct_sparse = wct_sparse
            
def main():
    if len(sys.argv)==4:
        K = int(sys.argv[1])
        lam = float(sys.argv[2])
        seed = int(sys.argv[3])
    else:
        K = 20
        lam = 0.9
        seed = 50
    with open("dblp-extrasmall.txt") as f:
        lines = f.readlines()
    #lines_prop = Doc(lines)
    docs = [line.strip().split() for line in lines]
    del(lines)
    docs_prop = Doc(docs)
    del(docs)
    print("Initialization!")
    pw_theta,pi_dj,pw_b=init(docs_prop,K,lam,seed)
    print("EM!")
    iteration, loglist,changelist,pw_theta = emstep(pw_theta,pi_dj,pw_b,docs_prop,K,lam)
    title = "K = "+str(K)+", lam = "+str(lam)+" for seed = "+str(seed)
    vcb_idx = docs_prop.vcb_idx
    whole_wl = get_topicwords(10, K, pw_theta ,vcb_idx)
    pl_log(loglist,changelist,title)
    return iteration, loglist,changelist,pw_theta,whole_wl

if __name__ == '__main__':
    iteration, loglist,changelist,pw_theta,whole_wl=main()
    