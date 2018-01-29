# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:33:55 2017

@author: evaljy
"""

########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log
import json

# Unknown word token
UNK = 'UNK'
def bLog(v):#inverse bounded log
    try:
        return log(v)
    except:
        return float("-inf")

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]
    def __str__(self):
        return '<'+self.word+' '+self.tag+'> '
    def __repr__(self):
        return '<'+self.word+' '+self.tag+'> '

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        with open("data/train.json") as fin:
          data = json.load(fin)
        words = list(a["words"] for a in data)
        pos_tags = list(a["pos_tags"] for a in data)
        return words, pos_tags
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                for w in sentence:
                    self.counts[w]+=1
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print( "Error: unlabeled data file", inputFile, "does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script  
### End file I/O methods ###  

    # Constructor
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
        

    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        words,pos_tags = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        print ("TRAIN\n--------------------------")
        for sentence,sentence2 in zip(words,pos_tags):
            #Use init probability instead of <s>
            self.init[sentence2[0]]+=1
            for idx, word in enumerate(sentence[1:]):
                i=idx+1
                t_i=sentence2[i-1]
                t_j=sentence2[i]
                w_j=sentence[i]
                if self.counts[w_j]<self.minFreq:
                    w_j=UNK
                self.tcounts[t_i][t_j]+=1
                self.wcounts[t_i][w_j]+=1
                
            
        for tag in self.init:
            self.init[tag]=float(self.init[tag])/len(self.init)

        vocab = self.tcounts.keys()
        self.vocab = vocab
        for t_i in vocab:
            total = float(sum(self.tcounts[t_i].values()))
            for t_j in vocab:#tcounts[t_i]:
                self.trans[t_i][t_j]=bLog(self.tcounts[t_i][t_j]/(total+len(vocab)))
            total = float(sum(self.wcounts[t_i].values()))
            for w_j in self.wcounts[t_i]:
                self.emit[t_i][w_j]=bLog(self.wcounts[t_i][w_j]/total)

    # Given an unlabeled corpus in testFile, output the viter tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):    
        data = self.readUnlabeledData(testFile)
        file=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            
            print(file, senString.rstrip())

    # Given a list of words, runs the viter algorithm and returns a list containing the sequence of tags 
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        for i, w in enumerate(words):
            if self.counts[w]<self.minFreq:
                words[i]=UNK

        # Initialization
        for t in self.vocab:
            w = words[0]
            self.viter[w][t]=self.init[t]+self.emit[t][w]
        # Recursion
        for idx, w in enumerate(words[1:]):
            i=idx+1
            for t in self.vocab:#self.emit.keys():
                self.viter[w][t]=float("-inf")
                for t_ in self.vocab:# Find max
                    tmp = self.viter[words[i-1]][t_]+self.trans[t_][t]
                    #print self.viter[words[i-1]][t_], self.trans[t_][t], tmp, t_
                    
                    if tmp > self.viter[w][t]:
                        self.viter[w][t]=tmp
                        self.backpointer[w][t]=t_
                    self.viter[w][t]+=self.emit[t][w]

        t_max = "NULL"
        vit_max = float("-inf")
        
        for t in self.vocab:
            w=words[-1]
            if self.viter[w][t] > vit_max:
                t_max = t
                vit_max = self.viter[w][t]     
        tags = ['']*len(words)
        i=len(words)-1
        
        while i>=0:
            tags[i]=t_max
            t_max = self.backpointer[words[i]][t_max]
            i-=1
        return tags

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('data/train.json')
    tagger.test('data/test_0', 'out.txt')