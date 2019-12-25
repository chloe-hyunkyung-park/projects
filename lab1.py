#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NLP lab1 text classification
# Registration No. 180228768
# Mon 18 Feb 2019

import sys, glob, os, re
import random
from collections import Counter
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

# The class of each textfile 
class Text:
    # original text content
    text = "" 
    # { n-gram : occurences }
    uni_tokens = dict() 
    bi_tokens = dict() 
    tri_tokens = dict() 
    # label (-1 : neg, 1 : pos)
    label = 0
    # Constructor
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.set_uni_tokens()
        self.set_bi_tokens()
        self.set_tri_tokens()
    def set_uni_tokens(self):
        tokens = list()                                                                                                                                                                                                                   
        tokens.extend( re.sub("[^\w']", " ", self.text).split() )
        stopwords = [
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 
                'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 
                'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don', 'down', 'during', 
                'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 
                'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 
                'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 
                'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 
                'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 
                't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 
                'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
                'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 
                'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves']
        # Consider only words without punctuations, spaces, even numbers
        # Remove stop-words
        tokens = [t for t in tokens if t.isalpha() and t not in stopwords]
        self.uni_tokens = dict(Counter(tokens))
    def set_bi_tokens(self):
        terms = list(self.uni_tokens.keys())
        bi_terms = list(zip(terms, terms[1:]))
        tokens = list(map(lambda x: ' '.join(x), bi_terms ))
        self.bi_tokens = dict(Counter(tokens))
    def set_tri_tokens(self):
        terms = list(self.uni_tokens.keys())
        tri_terms = list(zip(terms, terms[1:], terms[2:]))
        tokens = list(map(lambda x: ' '.join(x), tri_terms ))
        self.tri_tokens = dict(Counter(tokens))
        
# create initial weight dictionary depending on ngram
def get_iniweight(data, ngram):
    w = dict()
    tokens = set()
    # read all text instances
    for text in data:
        # get all keys from each text
        if ngram == 'uni' : tokens.update(set(text.uni_tokens))
        elif ngram == 'bi' : tokens.update(set(text.bi_tokens))
        elif ngram == 'tri' : tokens.update(set(text.tri_tokens))
    # initialize weights 0.0    
    for key in tokens : w[key] = 0.0
    return w
    
# input text instances and after learning, return average of w
def training(data, ngram, epoch):
    # initialize weight with {words of all texts :  0.0}
    w = get_iniweight(data, ngram)
    # sum of weights of each term to calculate average finally
    accum_w = w
    avg_w = w
    c = 1
    # evaluations of models with each updated weight
    e = list()
    # multiple passes over the training instances with epoch variable
    random.seed(365)
    for i in range(epoch):
        # shuffle the order of texts in train_data and test_data
        random.shuffle(data)
        for text in data:
            bow = dict()
            if ngram == 'uni' : bow = text.uni_tokens
            elif ngram == 'bi' : bow = text.bi_tokens
            elif ngram == 'tri' : bow = text.tri_tokens
            phi_x = list(bow.values())
            # extract only weight of term of this text's bow from w[c-1]
            vec_w = [ w[term] for term in bow ]
            y_hat = 1 if np.dot(vec_w, phi_x) >= 0 else -1
            if y_hat != text.label:
                for term in bow:
                    w[term] += text.label * bow[term]
            c += 1
        # sum of weights of eath term
        for term in w: accum_w[term] += w[term]
        # test model with updated weight for showing the learning progress in a graph
        e.append(testing(data, accum_w, ngram))
        
    # get the average of weights
    for term in accum_w: avg_w[term] = accum_w[term]/c
    return avg_w, e

# create graph of learning progress with four metrics depending on epoch
def plot_graph(evals):
    titles = ['uni', 'bi', 'tri']
    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(8,5))
    fig.suptitle('Learning progress of model with 4 metrics', fontsize=15, fontweight='bold')
    for j in range(len(evals)):
        e = evals[j]
        accuracy = []; precision = []; recall = []; f1_score = [];
        index = list(range(len(e)))
        for i in index:
            e_dic = e[i]
            accuracy.append(e_dic['accuracy'])
            precision.append(e_dic['precision'])
            recall.append(e_dic['recall'])
            f1_score.append(e_dic['f1'])
        # plotting
        axs[j].plot(index, accuracy, color = 'r', linewidth = 1.0, label = 'Accuracy' )
        axs[j].plot(index, precision, color = '#ff7f03', linewidth = 1.0, label = 'Precision' )
        axs[j].plot(index, recall, color = '#2ca02c', linewidth = 1.0, label = 'Recall' )
        axs[j].plot(index, f1_score, color = '#1f77b4', linewidth = 1.0, label = 'F1_Score' )
        axs[j].set_xlabel('Epoch')
        axs[j].set_ylabel('Probability')
        axs[j].set_title(titles[j] + '-gram')
        if j == 2:
            axs[j].legend()
    plt.show()

# input test_data, dictionary of weights
# and evaluate model(weights) with the metrics like accuracy, precision, recall, and F1         
def testing(data, w, ngram):
    # variables for calculating accuracy, precision, recall, and F1
    tp = 0; tn = 0; fp = 0; fn = 0;
    # read text instances for tesing model(weight)
    for text in data:
        bow = dict()
        if ngram == 'uni' : bow = text.uni_tokens
        elif ngram == 'bi' : bow = text.bi_tokens
        elif ngram == 'tri' : bow = text.tri_tokens
        score = 0.0
        y_hat = 1
        y = text.label # 1 or -1
        for term, c in bow.items():
            score += c*w[term]
        if score < 0 : y_hat = -1
        
        # set values of tp, tn, fp, fn
        # y == 1 (positive data)
        if y == 1 :
            # y == y_hat ( true ) 
            if y == y_hat : tp += 1
            # y != y_hat ( false )
            else : fp += 1
        # y == -1 (negative data)
        else :
            if y == y_hat :  tn += 1
            else : fn += 1
    # evaluation            
    e = dict()
    e[ 'accuracy' ] = (tp + tn) / len(data)
    e[ 'precision' ] = pre = (tp) / (tp + fp)
    e[ 'recall' ] = rec = (tp) / (tp + fn)
    e[ 'f1' ] = (2 * pre * rec) / (pre + rec)
    return e

# extract the most positively/negatively-weighted features for pos/neg class
def get_10features(w):
    ordered_w = sorted(w.items(), key = lambda k : k[1])
    # get 10 larger ones as the most positively-weighted features
    neg = ordered_w[:10]
    # get 10 lower ones as the most negatively-weighted features
    pos = ordered_w[-10:]
    return pos, neg
    
# input list of text files and return instances of Text class
def read_data(text_files, label):
    # list of text instances
    texts = list()
    for f in text_files:
        # read original text
        ori_text = open(f, 'r').read()
        texts.append( Text(ori_text, label) )
    return texts    

def load_data(path):
    train_data = list()
    test_data = list()
    pos_texts = glob.glob(os.path.join(path, 'txt_sentoken/pos/*.txt'))
    neg_texts = glob.glob(os.path.join(path, 'txt_sentoken/neg/*.txt'))
    # 1600 texts (800 pos, 800 neg) in train_data for training model
    train_data.extend(read_data(pos_texts[:800], 1))
    train_data.extend(read_data(neg_texts[:800], -1))
    # 400 texts (200 pos, 200 neg) in test_data for testing model
    test_data.extend(read_data(pos_texts[800:], 1))
    test_data.extend(read_data(neg_texts[800:], -1))
    return (train_data, test_data)

if __name__  == '__main__':
    # load text data
    path = os.path.join( os.getcwd(), sys.argv[1] )
    # get 1600 text instances for training and 400 text instances for testing
    train_data, test_data = load_data(path)
    n_gram = ['uni', 'bi', 'tri']
    
    # --------------------------------------------------------------------------
    # 'uni'
    # training(train_data, n_gram, epoch)
    w_uni, e_uni = training(train_data, n_gram[0], 10)
    # testing(train_data, weight, n_gram)
    # evaluation of uni-gram model with average of weights
    eval_uni = testing(train_data, w_uni, n_gram[0])
    # get top 10 positively/negatively weights
    pos10_uni, neg10_uni = get_10features(w_uni)
    print('-' * 50)
    print('[ uni-gram ]\n')
    pprint(eval_uni)
    print('\n[ Top 10 positively weigths ] \n\n', pos10_uni)
    print('\n[ Top 10 negatively weigths ] \n\n', neg10_uni, '\n', '-' * 50)

    # --------------------------------------------------------------------------
    # 'bi'
    w_bi, e_bi = training(train_data, n_gram[1], 10)
    # evaluation of bi-gram model with average of weights
    eval_bi = testing(train_data, w_bi, n_gram[1])
    pos10_bi, neg10_bi = get_10features(w_bi)
    print('[ bi-gram ]\n')
    pprint(eval_bi)
    print('\n[ Top 10 positively weigths ] \n\n', pos10_bi, '\n')
    print('\n[ Top 10 negatively weigths ] \n\n', neg10_bi, '\n', '-' * 50)

    # --------------------------------------------------------------------------
    # 'tri'
    w_tri, e_tri = training(train_data, n_gram[2], 10)
    # evaluation of tri-gram model with average of weights
    eval_tri = testing(train_data, w_tri, n_gram[2])
    pos10_tri, neg10_tri = get_10features(w_tri)
    print('[ tri-gram ]\n')
    pprint(eval_tri)
    print('\n[ Top 10 positively weigths ] \n\n', pos10_tri, '\n')
    print('\n[ Top 10 negatively weigths ] \n\n', neg10_tri, '\n', '-' * 50)
    
    # --------------------------------------------------------------------------
    # showing the learning progress in a graph
    plot_graph([e_uni, e_bi, e_tri])
