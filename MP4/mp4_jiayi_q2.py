# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:28:06 2017

@author: evaljy
"""
import json
import itertools
from collections import defaultdict
from collections import Counter
import sys

def viterbi(observations, start_probability, transition_probability, emission_probability,vocab):
    #save results
    result_m = [{}]
    states = tuple(start_probability.keys())
    
    #change low frequency words to UNK
    new_observations = []
    for i in range(len(observations)):
        if observations[i] in vocab:
            new_observations.append(observations[i])
        else:            
            new_observations.append('UNK')
    observations = new_observations
    
    #initialization
    for state in states:
        result_m[0][state] = (start_probability[state]*emission_probability[state][new_observations[0]],None) # 把第一个观测节点对应的各状态值计算出来
    #calculate
    for t in range(1,len(observations)):
        result_m.append({})  
        for state in states:
            result_m[t][state] = max([(result_m[t-1][s0][0]*transition_probability[s0][state]*emission_probability[state][new_observations[t]],s0) for s0 in states])
            
    return result_m 

def compare(result,tagging_true):
    #comparision between two list
    score = 0
    for result_i,tagging_true_i in zip(result,tagging_true):
        if result_i==tagging_true_i:
            score+=1
    return score

def most_common(hmm):
    emit = hmm.emit
    common = {}
    for i in emit.keys():
        temp = Counter(emit[i])
        common[i] = temp.most_common(10)
    return common 

class HMM:
    def __init__(self, minFreq=0): 
        self.minFreq = minFreq
        self.word_counts = defaultdict(int)
        self.tcounts = defaultdict(lambda : defaultdict(lambda: 1))
        self.wcounts = defaultdict(lambda : defaultdict(int))
        self.trans = defaultdict(lambda : defaultdict(float))
        self.emit = defaultdict(lambda : defaultdict(float))
        self.init = defaultdict(float)

    def train(self):
        print ("Read training data\n--------------------------")
        with open("data/train.json") as fin:
            data = json.load(fin)
        words = list(a["words"] for a in data)
        pos_tags = list(a["pos_tags"] for a in data)
        self.vocab = list(set(itertools.chain.from_iterable(words)))
        self.word_counts = Counter(item for sublist in words for item in sublist)
        print ("Begin training\n--------------------------")
        # calculate counts
        for word,pos_tag in zip(words,pos_tags):
            self.init[pos_tag[0]]+=1
            self.wcounts[pos_tag[0]][word[0]]+=1
            for idx, wordi in enumerate(word[1:]):
                t_i=pos_tag[idx]
                t_j=pos_tag[idx+1]
                w_j=word[idx+1]
                if self.word_counts[w_j]<=self.minFreq:
                    w_j='UNK'
                self.tcounts[t_i][t_j]+=1
                self.wcounts[t_j][w_j]+=1
        #normallization           
        for tag in self.init:
            self.init[tag]=float(self.init[tag])/len(words)
            
        tags = self.tcounts.keys()
        for t_i in tags:
            total = sum(self.tcounts[t_i].values())
            for t_j in self.tcounts[t_i]:
                self.trans[t_i][t_j]=self.tcounts[t_i][t_j]/total
            total = sum(self.wcounts[t_i].values())
            for w_j in self.wcounts[t_i]:
                self.emit[t_i][w_j]=self.wcounts[t_i][w_j]/total
        print("Finished training!\n--------------------------")
    
    def test(self,testfile):
        #read testfile
        file = open(testfile, "r") # open the input file in read-only mode
        print ("Read testing data\n--------------------------")
        words_test = [];
        for line in file:
            sentence = line.split() # split the line into a list of words
            words_test.append(sentence) # append this list as an element to the list of sentence
        #viterbi
        print ("Begin Viterbi\n--------------------------")
        #value initialization
        start_probability = self.init
        transition_probability = self.trans
        emission_probability = self.emit
        vocab = list(self.word_counts.keys())
        
        results = []
        for word_test in words_test:
            observations = tuple(word_test)
            result_m = viterbi(observations,
                     start_probability,
                     transition_probability,
                     emission_probability,vocab)
            result = []
            for resulti in result_m:
                result.append(max(resulti, key=resulti.get))
            results.append(result)
        print("Finished Viterbi!\n--------------------------")
        return results
    
    def score(self,testfile,checkfile):
        results = self.test(testfile)
        file = open(checkfile, "r") # open the input file in read-only mode
        print ("Read labled test data\n--------------------------")
        taggings_test = [];
        for line in file:
            sentence = line.split() # split the line into a list of words
            taggings_test.append(sentence) # append this list as an element to the list of sentence
        score = 0
        num = 0
        print ("Calculate accuracy\n--------------------------")
        for result,tagging_test in zip(results,taggings_test):
            score += compare(result,tagging_test)
            num += len(result)
        accuracy = score/num
        return accuracy, results
            
def write_in_results(file,results):
    f = open(file,'w')
    for line in results:
        for tag in line:
            f.write(tag)
            f.write(' ')
        f.write('\n')
    f.close() 

def write_in_common(file,common):
    f = open(file,'w')
    for tag in common.keys():
        f.write(tag)
        f.write('\n')
        for word in common[tag]:
            f.write(word[0])
            f.write(' ')
            f.write(str(word[1]))
            f.write('\n')
        f.write('\n')
    f.close()     
        
def main():
    minFreq = sys.argv[1]
    hmm = HMM(int(minFreq))
    hmm.train()
    #file = "test_1"
    file = sys.argv[2]
    testfile = "data/" + file
    checkfile = "data/" + file+"_tagging"
    #get the prediction results and compare them with the true value
    accuracy, results = hmm.score(testfile,checkfile)
    #find the most common words over each tag
    common = most_common(hmm)
    #generate a file to save
    results_file_path = file+"_results.txt"
    common_file_path = file+"_common.txt"
    write_in_results(results_file_path,results)
    write_in_common(common_file_path,common)
    print("The initial start probabilities are:")
    for key in hmm.init.keys():
        print(key," ",hmm.init[key])
    print("The top 10 words with the highest output probabilities for each POS tag for",testfile," are: ")
    for key in common.keys():
        print(key)
        print(common[key])
    #print accuracy
    print("\n The whole accuracy for ",testfile," is ",accuracy)

    return results,hmm,common

if __name__ == '__main__':
    results,hmm,common = main()
    
