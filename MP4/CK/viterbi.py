from collections import defaultdict
from collections import deque

import numpy as np
import json

class Viterbi():
    def __init__(self,threshold=0):
        self.threshold = threshold
        self.single_tag = defaultdict(int) # calculate p(word|tag)
        self.previous_tag = defaultdict(int) # calculate p(tag|previous_tag) previous tag use this count
        self.single_word = defaultdict(int)
        self.tag_tag = defaultdict(int)
        self.word_tag = defaultdict(int)
        self.initial = defaultdict(float)
        self.transition = defaultdict(float)
        self.emission = defaultdict(float)

    def read_data(self,path):

        with open(path) as f:
            data = json.load(f)
            # print(data[121]['words'])
            # print(data[121]['pos_tags'])
        self.vocab = defaultdict(int)
        self.tagset = set()
        self.len_sen = len(data)
        # print(len(data))
        for i in range(len(data)):
            words = data[i]['words']
            tags = set(data[i]['pos_tags'])
            self.tagset.update(tags)
            for j in range(len(words)):
                word = words[j]#.lower()
                self.vocab[word] += 1
                #data[i]['words'][j] = word # treat upper case words the same as lower case words
        vocabulary = set(self.vocab.keys())
        for word in vocabulary:

            if self.vocab[word] <= self.threshold:
                self.vocab['UNK'] += self.vocab[word]
                del self.vocab[word]
        self.vocabulary = set(self.vocab.keys())
        self.tag_list = list(self.tagset)

        print(len(self.vocabulary))

        return data


    def train(self,train):
        for i in range(len(train)):

            words = train[i]['words']
            tags = train[i]['pos_tags']

            for j in range(len(words)):
                word = words[j]
                tag = tags[j]
                if word not in self.vocabulary:
                    word = 'UNK'

                if j == 0:
                    previous_tag = '#s'
                    self.single_tag[tag] += 1
                    # self.previous_tag[previous_tag] += 1
                    self.single_word[word] += 1
                    self.word_tag[(tag, word)] += 1
                    self.tag_tag[(previous_tag, tag)] += 1
                else:
                    previous_tag = tags[j-1]
                    self.single_tag[tag] += 1
                    self.previous_tag[previous_tag] += 1
                    self.single_word[word] += 1
                    self.word_tag[(tag,word)] += 1
                    self.tag_tag[(previous_tag,tag)] += 1

        # print(self.word_tag)
        # print(self.tag_tag)
        # add 1 smoothing to make sure transition prob != 0
        for tag1 in self.tagset:

            for tag2 in self.tagset:
                # self.single_tag[tag1] += 1
                self.tag_tag[(tag2,tag1)] += 1
                # self.single_tag[tag2] += 1

                self.previous_tag[tag2] += 1

        self.cal_initial()
        self.cal_emission()
        self.cal_transition()



    def cal_initial(self):
        for tag in self.tagset:
            self.initial[tag] = self.tag_tag[('#s',tag)]/self.len_sen
        print('initial:',self.initial)
        # print(sum(self.initial.values()))




    def cal_transition(self):
        for previous_tag in self.tagset:
            for tag in self.tagset:
                self.transition[(previous_tag,tag)] = self.tag_tag[(previous_tag,tag)]/self.previous_tag[previous_tag]




    def cal_emission(self):
        for word in self.vocabulary:
            for tag in self.tagset:
                self.emission[(tag,word)] = self.word_tag[(tag,word)]/self.single_tag[tag]


    def top_10_words(self):

        for tag in self.tagset:
            temp = []
            for word in self.vocabulary:
                temp.append((word,self.emission[(tag,word)]))

            ordered_list = sorted(temp,key = lambda x: x[1],reverse=True)
            print('tag:',tag,ordered_list[0:10])
            ###
            # this is used for check emission prob sum = 1, not used in assignment
            # s = 0
            # for t in ordered_list:
            #     s += t[1]
            # print('tag:',tag,s)
            ###

    def viterbi(self,sen):

        # initial:
        trellis = np.empty((len(self.tagset),len(sen)),dtype=object)
        # from second to the last word
        for i in range(len(sen)):

            word = sen[i]#.lower()

            if word not in self.vocabulary:
                print(word)
                word = 'UNK'
                

            for j in range(len(self.tag_list)):
                # print(word)
                # print('j:',j)
                temp = []
                tag = self.tag_list[j]

                if i == 0:
                    if self.emission[(tag,word)] == 0:
                        prob = 0
                    else:
                        prob = np.log(self.initial[tag])+np.log(self.emission[(tag,word)])

                    if prob == 0:
                        trellis[j, i] = (float('-inf'), float('-inf'))
                    else:
                        trellis[j,i] = (prob,float('-inf'))

                else:
                    for k in range(len(self.tag_list)):
                        # print('k:',k)
                        previous_tag = self.tag_list[k]
                        transition = self.transition[(previous_tag,tag)]

                        last = trellis[k,i-1][0]

                        if last == float('-inf'):
                            temp.append(float('-inf'))
                        else:
                            # print(last)
                            temp.append(np.log(transition)+last)
                        # print(temp)
                    back_pointer = np.argmax(temp)
                    if self.emission[(tag,word)] == 0:
                        prob = 0
                    else:
                        prob = np.max(temp)+np.log(self.emission[(tag,word)])
                    #print('tag:', tag, 'word:', word, 'prob:', prob)
                    if prob == 0:
                        trellis[j, i] = (float('-inf'), float('-inf'))
                    else:
                        trellis[j, i] = (prob, back_pointer)
                    # print(trellis)
        #print(self.tag_list)
        #print(trellis)
        # back track

        d = deque()
        temp = []
        for i in range(len(sen)-1,0,-1):
            # print('backward:',i)
            if i == len(sen)-1:
                for j in range(len(self.tag_list)):
                    temp.append(trellis[j,i][0])
                idx = np.argmax(temp)
                back_pointer = trellis[idx, i][1]
                d.appendleft(self.tag_list[idx])
            else:
                d.appendleft(self.tag_list[back_pointer])
                back_pointer = trellis[back_pointer,i][1]

        d.appendleft(self.tag_list[back_pointer])
        return d



    def test(self,test_path,prediction_path):
        output = open(prediction_path,'w')
        f = open(test_path,'r')

        for line in f:

            sen = line.split()

            tag_sequence = self.viterbi(sen)

            for i in range(len(tag_sequence)):
                tag = tag_sequence[i]
                if i == len(tag_sequence)-1:
                    output.write(tag)
                    output.write('\n')
                else:
                    output.write(tag+' ')

        f.close()
        output.close()




    def evaluation(self,prediction_path,true_label_path):
        with open(prediction_path,'r') as predict:
            predictions = predict.readlines()
        with open(true_label_path,'r') as f:
            true_labels = f.readlines()
        correct_label = 0
        correct_sen = 0
        count_label = 0
        count_sen = 0
        for i in range(len(predictions)):

            prediction = predictions[i]
            truths = true_labels[i]
            flag = 0
            pred = prediction.split()
            truth = truths.split()

            for j in range(len(pred)):
                if pred[j] == truth[j]:
                    correct_label += 1
                else:
                    flag = 1
                count_label += 1
            if flag == 0:
                correct_sen += 1
                #print(i)
            count_sen += 1

        print('label acc:',float(correct_label/count_label))

        print('sentence acc:', float(correct_sen/count_sen))





if __name__ == "__main__":
    
    path = 'data/train.json'
    test_path_0 = 'data/test_0'
    '''
    Vit = Viterbi(0)
    train = Vit.read_data(path)
    Vit.train(train)
    Vit.top_10_words()
    prediction_path_0 = 'output_0.txt'
    Vit.test(test_path_0,prediction_path_0)

    true_label_path = 'data/test_0_tagging'
    Vit.evaluation(prediction_path_0,true_label_path)
'''
    test_path_1 = 'data/test_1'
    Vit2 = Viterbi(1)
    train = Vit2.read_data(path)
    Vit2.train(train)
    Vit2.top_10_words()
    prediction_path_1 = 'output_1.txt'
    Vit2.test(test_path_1,prediction_path_1)

    true_label_path1 = 'data/test_1_tagging'
    Vit2.evaluation(prediction_path_1, true_label_path1)





