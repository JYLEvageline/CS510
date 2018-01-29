# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:28:06 2017

@author: evaljy
"""
import json
import itertools
import numpy as np
from collections import defaultdict
from math import log
import os
import sys
'''
def train():
    # Compute the start probabilities, transition probabilities and output probabilities 
    with open("data/train.json") as fin:
      data = json.load(fin)
    words = list(a["words"] for a in data)
    pos_tags = list(a["pos_tags"] for a in data)
    words_list =  list(set(itertools.chain.from_iterable(words)))
    pos_tags_list =  list(set(itertools.chain.from_iterable(pos_tags)))
    wt = np.zeros([len(words_list),len(pos_tags_list)])
    i=0
    # p(w|t)
    for word,pos_tag in zip(words,pos_tags):
        i +=1
        print(i)
        for word_t, pos_tag_t in zip(word,pos_tag):
            word_index = words_list.index(word_t)
            pos_index = pos_tags_list.index(pos_tag_t)
            wt[word_index,pos_index]+=1
    wt_final = np.divide(wt,wt.sum(axis = 0))
    #p(t|t)
    tt = np.zeros([len(pos_tags_list),len(pos_tags_list)])
    t = np.zeros(len(pos_tags_list))
    for pos_tag in pos_tags:
        pos_tag_former = pos_tag[0]
        index_begin = pos_tags_list.index(pos_tag_former)
        t[index_begin]+=1
        for i in range(1,len(pos_tag)):
            pos_tag_latter = pos_tag[i]
            index_former = pos_tags_list.index(pos_tag_former)
            index_latter = pos_tags_list.index(pos_tag_latter)
            tt[index_former,index_latter]+=1
    t_final = np.divide(t,len(pos_tags))
    tt_final = np.divide(tt,tt.sum(axis = 0))
    return t_final, tt_final, wt_final
'''


def viterbi(obs, states, start_p, trans_p, emit_p):

  result_m = [{}] # 存放结果,每一个元素是一个字典，每一个字典的形式是 state:(p,pre_state)
                  # 其中state,p分别是当前状态下的概率值，pre_state表示该值由上一次的那个状态计算得到
  for s in states:  # 对于每一个状态
    result_m[0][s] = (start_p[s]*emit_p[s][obs[0]],None) # 把第一个观测节点对应的各状态值计算出来

  for t in range(1,len(obs)):
    result_m.append({})  # 准备t时刻的结果存放字典，形式同上

    for s in states: # 对于每一个t时刻状态s,获取t-1时刻每个状态s0的p,结合由s0转化为s的转移概率和s状态至obs的发散概率
                     # 计算t时刻s状态的最大概率，并记录该概率的来源状态s0
                     # max()内部比较的是一个tuple:(p,s0),max比较tuple内的第一个元素值
        if obs[t] in emit_p[s]:
            result_m[t][s] = max([(result_m[t-1][s0][0]*trans_p[s0][s]*emit_p[s][obs[t]],s0) for s0 in states])
     # else:
      #    result_m[t][s] = 1
  return result_m 

def compare(result,tagging_true):
    #comparision between two list
    score = 0
    for result_i,tagging_true_i in zip(result,tagging_true):
        if result_i==tagging_true_i:
            score+=1
    return score

class HMM:
    def __init__(self, unknownWordThreshold=0): 
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        self.counts = defaultdict(int)
        ### Initialize the rest of your data structures here ###
        self.tcounts = defaultdict(lambda : defaultdict(lambda: 1)) #counts[i][j]=C(t_i.t_j). WITH ADD-ONE Smoothing
        self.wcounts = defaultdict(lambda : defaultdict(int)) #counts[i][j]=C(w_j.t_i)
        self.emit = defaultdict(lambda : defaultdict(float))
        self.trans = defaultdict(lambda : defaultdict(float))
        self.viter = defaultdict(lambda : defaultdict(float))
        self.backpointer = defaultdict(lambda : defaultdict(int))
        self.init = defaultdict(float)
        self.backpointer = defaultdict(lambda : defaultdict(str))

    def train(self):
        with open("data/train.json") as fin:
            data = json.load(fin)
        words = list(a["words"] for a in data)
        pos_tags = list(a["pos_tags"] for a in data)
        print ("TRAIN\n--------------------------")
        for word,pos_tag in zip(words,pos_tags):
            #Use init probability instead of <s>
            self.init[pos_tag[0]]+=1
            self.wcounts[pos_tag[0]][word[0]]+=1
            for idx, wordi in enumerate(word[1:]):
                i=idx+1
                t_i=pos_tag[i-1]
                t_j=pos_tag[i]
                w_j=word[i]
                self.tcounts[t_i][t_j]+=1
                self.wcounts[t_j][w_j]+=1
                           
        for tag in self.init:
            #self.init[tag]=bLog(float(self.init[tag])/len(self.init))
            self.init[tag]=float(self.init[tag])/len(words)
            
        vocab = self.tcounts.keys()
        self.vocab = vocab
        for t_i in vocab:
            total = sum(self.tcounts[t_i].values())
            for t_j in self.tcounts[t_i]:#tcounts[t_i]:
                #self.trans[t_i][t_j]=bLog(self.tcounts[t_i][t_j]/(total+len(vocab)))
                self.trans[t_i][t_j]=self.tcounts[t_i][t_j]/total
            total = sum(self.wcounts[t_i].values())
            for w_j in self.wcounts[t_i]:
                #self.emit[t_i][w_j]=bLog(self.wcounts[t_i][w_j]/total)  
                self.emit[t_i][w_j]=self.wcounts[t_i][w_j]/total
    
    def test(self,testfile):
        file = open(testfile, "r") # open the input file in read-only mode
        print ("Read test data\n--------------------------")
        words_test = [];
        for line in file:
            sentence = line.split() # split the line into a list of words
            words_test.append(sentence) # append this list as an element to the list of sentence
        print ("Viterbi begin\n--------------------------")
        start_probability = self.init
        transition_probability = self.trans
        emission_probability = self.emit
        states = tuple(start_probability.keys())
        results = []
        for word_test in words_test:
            observations = tuple(word_test)
            result_m = viterbi(observations,
                     states,
                     start_probability,
                     transition_probability,
                     emission_probability)
            result = []
            for resulti in result_m:
                result.append(max(resulti, key=resulti.get))
            results.append(result)
        return results
    
    def score(self,testfile,checkfile):
        results = self.test(testfile)
        file = open(checkfile, "r") # open the input file in read-only mode
        print ("Read labeld test data\n--------------------------")
        taggings_test = [];
        for line in file:
            sentence = line.split() # split the line into a list of words
            taggings_test.append(sentence) # append this list as an element to the list of sentence
        score = 0
        num = 0
        for result,tagging_test in zip(results,taggings_test):
            score += compare(result,tagging_test)
            num += len(result)
        accuracy = score/num
        return accuracy, results
            

    
        
def main():
    hmm = HMM()
    hmm.train()
    testfile = "data/test_0"
    checkfile = "data/test_0_tagging"
    accuracy, results = hmm.score(testfile,checkfile)
    print(accuracy)
    return results

if __name__ == '__main__':
    results = main()
    
