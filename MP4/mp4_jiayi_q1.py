# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:43:19 2017

@author: evaljy
"""
class HMM:
    def __init__(self):
        #trans
        self.trans = {"V":{"V":0.6,"N":0.4},
                     "N":{"V":0.5,"N":0.5}}
        #emit
        self.emit = {"V":{"hello":0.6,"world":0.1,"print":0.2,"line":0.1},
                     "N":{"hello":0.1,"world":0.6,"print":0.1,"line":0.2}}
        #init
        self.init = {"V":0.5,
                     "N":0.5}
    def forward(self, obs):
        self.fwd = [{}]     
        for key in self.trans.keys():
            self.fwd[0][key] = self.init[key] * self.emit[key][obs[0]]
            #print(self.fwd[0][y])
        for t in range(1, len(obs)):
            self.fwd.append({}) 
            for key in self.trans.keys():
                self.fwd[t][key] = sum((self.fwd[t-1][y0] * self.trans[y0][key] * self.emit[key][obs[t]]) for y0 in self.trans.keys())
                #print(self.fwd[t][y])
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.trans.keys())
        return prob
    
    def backward(self, obs):
        self.bwk = [{} for i in range(len(obs))]
        T = len(obs)
        for key in self.trans.keys():
            self.bwk[T-1][key] = 1
            #print(self.bwk[T-1][y])
        for t in reversed(range(T-1)):
            for key in self.trans.keys():
                self.bwk[t][key] = sum((self.bwk[t+1][y1] * self.trans[key][y1] * self.emit[y1][obs[t+1]]) for y1 in self.trans.keys())
                #print(self.bwk[t][y])
        prob = sum((self.init[key]* self.emit[key][obs[0]] * self.bwk[0][key]) for key in self.trans.keys())
        return prob

if __name__ == '__main__':
    obs = "print  line  hello  world"
    obs = obs.split()
    q1 = HMM()
    print("Forward Algorithm\n--------------------------")
    prob = q1.forward(obs)
    for i in range(len(q1.fwd)):
        for key in q1.fwd[i].keys():
            print("\\alpha_",i+1,"(",key,") = ",q1.fwd[i][key],"\\\\")
    print("P(print line hello world) = ",prob)
    print("Backward Algorithm\n--------------------------")
    prob = q1.backward(obs)
    for i in range(len(q1.bwk)):
        for key in q1.bwk[i].keys():
            print("\\beta_",i+1,"(",key,") = ",q1.bwk[i][key],"\\\\")
    print("P(print line hello world) = ",prob)