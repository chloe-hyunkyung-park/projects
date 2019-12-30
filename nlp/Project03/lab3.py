#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NLP lab3 Named Entity Recognition with the Structured Perceptron
# Registration No. 180228768
# Mon 18 Mar 2019

import sys, glob, os, re, copy
import random
from collections import Counter
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Loading the data
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

'''
functions that extract and transform a sentence into different feature types(Phi() in Perceptron)
in the form of Python dictionaries
'''

# 1. Current word-Current label

# 1-1) Function1 - get_cwcl(corpus)
#         input : c = [s1, s2, ..], s1 = [(x1, y1), .., (xn, yn)]
#         output : {'x1_y1' : 103, ..}
def get_cwcl(corpus):
    cwcl = dict()
    for sen in corpus:
        for t in sen:   # t : (word, tag), t means tuple
            word = t[0]; tag = t[1];
            token = '_'.join([word, tag])
            if token in cwcl : cwcl[token] += 1
            else : cwcl[token] = 1
    # discard some features with small frequency
    freq_limit = 3
    for k in list(cwcl.keys()):
        if cwcl[k] < freq_limit: del cwcl[k]
    return cwcl

# 1-2) Function2 - phi_1(x, y, cw_cl_counts)
#         input : a sentence(words,x and tags,y)
#         output : dictionary with counts of the cw_cl_counts keys in the given sentence(x, y)
def phi_1(x, y, cwcl):
    feature = dict()
    for i in range(len(x)):
        word = x[i]; tag = y[i];
        token = '_'.join([word, tag])
        if token in cwcl: feature[token] = cwcl[token]
        else : feature[token] = 0
    return feature

# 2. Previous label-Current label, phi_2(x,y)
'''
    You need to write two functions following a similar two-step of phi_1(x,y) above.
    After you have the feature extraction functionality in place, you will need to combine phi_2 and phi_1
    to see whether the new feature type improves performance.
    This can be done by merging the two Python dictionaries you obtain for each feature type.
'''
# 2-1) Function1 - (corpus)
#         input : c = [s1, s2, ..], s1 = [(x1, y1), .., (xn, yn)]
#         output : {'previous tag_current tag' : freq. of the tag in Corpus', ..}
def get_ptct(corpus):
    ptct = dict()
    for sen in corpus :
        # extract only tags in a sentence
        tags = list(list(zip(*sen))[1])
        # Add the 'Nope' tag at the begining of each sentence
        tags.insert(0, 'Nope')
        for i in range(len(tags) - 1):
            # make tokens
            token = '_'.join([tags[i], tags[i+1]])
            if token in ptct : ptct[token] += 1
            else : ptct[token] = 1
    # discard some features with small frequency
    freq_limit = 3
    for k in list(ptct.keys()):
        if ptct[k] < freq_limit: del ptct[k]
    return ptct

# 2-2) Function2 - phi_2(x, y, ptct)
#         input : a sentence(words,x and tags,y)
#         output : dictionary with counts of the ptct keys in the given sentence(x, y)
def phi_2(x, y, ptct):
    feature = dict()
    tags = list(y)
    # Add the 'Nope' tag at the begining of each sentence
    tags.insert(0, 'Nope')
    for i in range(len(tags) - 1):
        # make tokens
        token = '_'.join([tags[i], tags[i+1]])
        if token in ptct: feature[token] = ptct[token]
        else : feature[token] = 0
    return feature

# [ Combine phi_2 and phi_1 ]
#       to see whether the new feature type improves performance.
#       This can be done by merging the two Python dictionaries you obtain for each feature type.
#       corpus can be train_data or test_data
def phi_mix(x, y, cwcl, ptct):
    # cwcl feature of one sentence
    s_cwcl = phi_1(x, y, cwcl)
    # ptct feature of one sentence
    s_ptct = phi_2(x, y, ptct)
    f_mix = s_cwcl.copy()   # start with x's keys and values
    f_mix.update(s_ptct)
    return f_mix

'''
3. Optional Bonus

    Add at least two more feature types(phi_3, phi_4) of your choice following the same approach as for phi_1
    and combine them with phi_1 and phi_2 to test if you obtain improved performance.
    Ideas : sub word features, previous/next words, label trigrams, etc,..
'''
# feature 3 : label trigrams
# add 'None' in both the beginning and end of the each sentence
def label_tri(corpus):
    trilabel = list()
    for sen in corpus :
        # extract only tags in a sentence
        tags = list(list(zip(*sen))[1])
        # Add the 'Nope' tag at the begining and end of each sentence
        tags.insert(0, 'Nope')
        tags.append('Nope')
        for i in range(len(tags) - 2):
            # make tokens
            token = '_'.join([tags[i], tags[i+1], tags[i+2]])
            trilabel.append(token)
    return Counter(trilabel)

def phi_3(x, y, trilabel):
    feature = dict()
    y = list(y)
    y.insert(0, 'Nope')
    y.append('Nope')
    for i in range(len(y)-2):
        token = '_'.join([y[i], y[i+1], y[i+2]])
#        print('token : ', token)
        if token in trilabel: feature[token] = trilabel[token]
        else : feature[token] = 0
    return feature

# feature 4 : previous word and current word
def get_pwcw(corpus):
    pwcw = dict()
    for sen in corpus :
        # extract only words in a sentence
        words = list(list(zip(*sen))[0])
        for i in range(len(words) - 1):
            # make tokens
            token = '_'.join([words[i], words[i+1]])
            if token in pwcw : pwcw[token] += 1
            else : pwcw[token] = 1
    # discard some features with small frequency
    freq_limit = 3
    for k in list(pwcw.keys()):
        if pwcw[k] < freq_limit: del pwcw[k]
    return pwcw

def phi_4(x, y, pwcw):
    feature = dict()
    words = list(x)
    #print('words : ', words)
    for i in range(len(words) - 1):
        # make tokens
        token = '_'.join([words[i], words[i+1]])
        #print('token : ', token)
        if token in pwcw:
            #print(pwcw[token])
            feature[token] = pwcw[token]
        else:
            feature[token] = 0
    return feature

# input : a sentence
# output : return Counter of 'tag_tag' tokens in a sentence
def tt_feat(sen):
    tokens = list()
    # extract only tags in a sentence
    y = list(list(zip(*sen))[1])
    y.insert(0, 'Nope')
    for i in range(len(y) - 1):
        tokens.append('_'.join([y[i], y[i+1]]))
    return Counter(tokens)

# input : a sentence
# output : return Counter of 'word_tag' tokens in a sentence
def wt_feat(sen):
    tokens = list()
    for x,y in sen:
        tokens.append('_'.join([x, y]))
    return Counter(tokens)

# find the only one tag of a word
def argmax_y(w, token, freq):
    x, y = token.split('_')
    labels = ['O', 'PER', 'LOC', 'ORG', 'MISC']
    # fine max of w*phi in this list
    find_max = dict()
    for l in labels:
        new_token = '_'.join([x,l])
        find_max[new_token] = 0.0
        if new_token not in w: w[new_token] = 0.0
        find_max[new_token] =  w[new_token]*freq
        #print('new_token : ', new_token, 'find_max : ', find_max)

    # find pred_y with max_value
    pred_y = token
    base = find_max[pred_y]
    for t, f in find_max.items():
        if f > base:
            pred_y = t

    return w, pred_y.split('_')[1]


#output : [('I', 'PER'), ('hospital', 'LOC')]
def predict(w, phi):
    y_hat = list()
    #print('phi : ', phi)
#    for token, freq in phi.items():
    for e in phi:
       # print('e : ', e)
        x, y = e.split('_')
         # predicted y of x for argmax w*phi(x,y)
        w, pred_y = argmax_y(w, e, phi[e])
       # print('y : ', y)
       # print('pred_y : ', pred_y)
        y_hat.append( (x, pred_y) )
    #print('y_hat : ', y_hat)
    return w, y_hat


# Implement the Structured Perceptron
def train(train_data, epoch):
    # Extract the features of train_data
    # cwcl : the features of the current_word and current_tag(label)
    cwcl = get_cwcl(train_data)
    # ptct : the features of the previous_tag and current_tag
    ptct = get_ptct(train_data)

    # model1
    # w_1 : the weight dictionary of cwcl
    # initialize weights 0.0 before training
    w_1 = dict.fromkeys(list(cwcl.keys()), 0.0)

    # model2
    # w_mix : the weight dictionary of cwcl and ptct
    # phi_mix : the mix of features of phi_1 and phi_2
    w_mix = dict.fromkeys(list(cwcl.keys()) + list(ptct.keys()), 0.0)

    # Learning
    # epoch is for multiple passes with randomised order and averaging.
    random.seed(365)
    for i in range(epoch):
        random.shuffle(train_data)
        for sen in train_data:
            x, y = zip(*sen)
            #print('sen : ', sen)

            ##### model1(m1),w_1
            feat_correct_m1 = wt_feat(sen)
            phi1 = phi_1(x, y, cwcl)
            #print('phi1 : ',  phi1)
            if(isinstance(phi1, dict)): w_1, y_hat_m1 = predict(w_1, phi1)
            else: break
            feat_predicted_m1 = wt_feat(y_hat_m1)
            #print('feat_correct_m1 : ', feat_correct_m1)
            #print('feat_predicted_m1 : ', feat_predicted_m1)
            if feat_correct_m1 != feat_predicted_m1:
                feat_diff = Counter(feat_correct_m1)
                feat_diff.subtract(feat_predicted_m1)
                # update w
                for k in feat_diff:
                    #print('feat_diff : ', feat_diff)
                    diff = feat_diff[k]
                    if diff != 0 and k in feat_predicted_m1:
                        w_1[k] += diff

            ##### model2(m2),w_mix
            feat_correct_m2 = tt_feat(sen)
            phimix = phi_mix(x, y, cwcl, ptct)
            #print('phimix : ',  phimix)
            if(isinstance(phimix, dict)) : w_mix, y_hat_m2 = predict(w_mix, phimix)
            else: break
            feat_predicted_m2 = wt_feat(y_hat_m2)
            #print('feat_correct_m2 : ', feat_correct_m2)
            #print('feat_predicted_m2 : ', feat_predicted_m2)
            if feat_correct_m2 != feat_predicted_m2:
                feat_diff = Counter(feat_correct_m2)
                feat_diff.subtract(feat_predicted_m2)
                # update w
                for k in feat_diff:
                    #print('feat_diff : ', feat_diff)
                    diff = feat_diff[k]
                    if diff != 0 and k in feat_predicted_m2:
                        w_mix[k] += diff
    return w_1, w_mix

def test(test_data, w_1, w_mix):
    y_true = list();
    #e_phi1 = list()
    #e_phimix = list()
    # cwcl : the features of the current_word and current_tag(label)
    cwcl = get_cwcl(train_data)
    # ptct : the features of the previous_tag and current_tag
    ptct = get_ptct(train_data)
    y_true = list()
    y_predicted_phi1 = list()
    y_predicted_phimix = list()

    scaler = StandardScaler()

    for sen in test_data:

        x, y = zip(*sen)
        phi1 = phi_1(x, y, cwcl)
        phimix = phi_mix(x, y, cwcl, ptct)
        y_true.append(sen)
        y_predicted_phi1.append(predict(w_1, phi1 ))
        y_predicted_phimix.append(predict(w_mix, phimix))
        print('y_true : \n', y_true)
        print('y_predicted_phi1 : \n', y_predicted_phi1)
        print('y_predicted_phimix : \n', y_predicted_phimix)
        return
#    y_ = yscaler_x.transform(y_true)
    f1_micro_phi1 = f1_score(y_true, y_predicted_phi1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    f1_micro_phimix = f1_score(y_true, y_predicted_phimix, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
#        e_phi1.append(f1_micro_phi1)
#        e_phimix.append(f1_micro_phimix)
    return f1_micro_phi1, f1_micro_phimix

# this function prints all 4 features of the sentence
def check_4features(corpus):
    random.seed(365)
    sen = corpus[random.randint(0, len(corpus))]
    x, y = zip(*sen)
    print('sentence : ')
    pprint(sen)
    # phi_1
    # cwcl : the features of the current_word and current_tag(label)
    cwcl = get_cwcl(train_data)
    phi1 = phi_1(x, y, cwcl)
    print('phi_1 : ')
    pprint(phi1)
    # phi_2
    # ptct : the features of the previous_label(tag) and current_label(tag)
    ptct = get_ptct(train_data)
    phi2 = phi_2(x, y, ptct)
    print('phi_2 : ')
    pprint(phi2)
    # phi_3
    # trilabel :
    trilabel = label_tri(train_data)
    phi3 = phi_3(x, y, trilabel)
    print('phi_3 : ')
    pprint(phi3)
    # phi_4
    # pwcw : the features of the previous_word and current_word
    pwcw = get_pwcw(train_data)
    phi4 = phi_4(x, y, pwcw)
    print('phi_4 : ')
    pprint(phi4)



if __name__  == '__main__':
    train_path = os.path.join( os.getcwd(), sys.argv[1] )
    test_path = os.path.join( os.getcwd(), sys.argv[2] )

    train_data = load_dataset_sents('train.txt')

    # check the four features of the sentence which is randomly picked
    #check_4features(train_data)

    epoch = 10
    # training the model with the structured perceptron and train_data
    w_1, w_mix = train(train_data, epoch)

    #test_data = load_dataset_sents('test.txt')
    #f1_phi1, f1_phimix = test(test_data, w_1, w_mix)
    #print("e_phi1 : ", f1_phi1)
    #print("e_phimix : ", f1_phimix)
