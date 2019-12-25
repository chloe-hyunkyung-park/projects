#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NLP lab2 Language modelling
# Registration No. 180228768
# 01  Mar 2019

import sys, os, re
import string
from collections import Counter

# Language Model class : include each sentence
class LM:
    # list of sentences  without <s>,</s>
    corpus = list()
    # list of sentences  with <s>,</s>
    sentences = list()
    # include <s> and </s> in each sentence
    tokens = list()
    # dictionary of uni, bi-gram with {token:counts of token}
    uni_tokens = dict()
    bi_tokens = dict()
    # the number of all uni, bi-gram
    uni_all = 0
    bi_all = 0
    smoothing = False
    def __init__(self, corpus, smoothing = False):
        self.corpus = corpus
        self.set_sentences()
        self.set_uni_tokens()
        self.set_bi_tokens()
    def set_sentences(self):
        for s in self.corpus:
            s = '<s> ' + s + ' </s>'
            self.sentences.append(s)
    def set_uni_tokens(self):
        # To create uni_tokens, we don't treat <s>,</s> as tokens. 
        # so, choose choose the list of sentences without <s>,</s> - "self.corpus", not "self.sentences"
        self.tokens = list()                                                     
        for s in self.corpus :
            l = s.lower().split()
            for w in l:
                # remove tokens of a punctuation and more than two connected punctuations like ';;', '----', etc
                if w in string.punctuation or re.match(r'(([^\w\s])\2+)', w):
                    continue
                self.tokens.append(w)
        # uni-tokens : dictionary of {token:counts}
        self.uni_tokens = dict(Counter(self.tokens))
        # uni_all : the number of all uni-tokens (not unique, including all tokens)
        self.uni_all = len(self.tokens)
    def get_prob_uni(self, questions, candidates):
        result = list()
        for c1, c2 in candidates:
            p1 = self.uni_tokens[c1]/self.uni_all
            p2 = self.uni_tokens[c2]/self.uni_all
            if p1 < p2 : result.append((c2, p2))
            elif p1 == p2 : 
                if p1 == 0: 
                    result.append(("ZERO probability", 0))
                else: result.append(("SAME probability", 0))
            else : result.append((c1, p1))
        return result
    def set_bi_tokens(self):
        # To create bi_tokens, we treat <s>,</s> as tokens, so choose "self.sentences", not "self.corpus"
        temp_tokens = list()
        for s in self.sentences :
            l = s.lower().split()
            for w in l:
                # remove tokens of a punctuation like '-','.',''','"','/',...
                if w not in string.punctuation:
                    # remove tokens of connected punctuations like '----', ';;;;;', '!?!?!?!', ...
                    if not re.match(r'(([^\w\s])\2+)', w):
                        temp_tokens.append(w)
        bi_terms = list(zip(temp_tokens, temp_tokens[1:]))
        for bi_t in bi_terms:
            if bi_t in self.bi_tokens : self.bi_tokens[bi_t] += 1
            else : self.bi_tokens[bi_t] = 1
        self.bi_all = len(bi_terms)
    def get_prob_bi(self, questions, candidates, smoothing = False):
        result = list()
        pre_next = self.get_pre_n_next(questions)
        num = len(questions)
        for i in range(num):  
            # c1 = candidates[i][0], c2 = candidates[i][1]
            # pre = pre_next[i][0], next = pre_next[i][1]
            c1 = candidates[i][0]
            c2 = candidates[i][1]
            pre_word = pre_next[i][0]
            next_word = pre_next[i][1]
            # with candidate 1
            # c1_pre : (pre, candidate1), c1_next : (candidate1, next)
            c1_pre = (pre_word, c1)
            c1_next = (c1, next_word)
            # with candidate 2
            # c2_pre : (pre, candidate2), c2_next : (candidate2, next)
            c2_pre = (pre_word, c2)
            c2_next = (c2, next_word)
            # calculate probability 
            p_c1 = 0; p_c2=0;
            
            # bigram with smoothing
            # P(x(n)|x(n-1)) = ( counts(x(n-1), x(n)) + 1 ) / ( counts(x(n-1)) + size_V)
            if smoothing:
                abs_v = self.bi_all
                ## c1 : candidate1
                # c1_pre not in self.bi_tokens:
                if c1_pre not in self.bi_tokens:
                    if c1_next not in self.bi_tokens:
                        p_c1 = (1/abs_v) * (1/abs_v)
                    else:
                        p_c1 = (1/abs_v) * (self.bi_tokens[c1_next] + 1)/((self.uni_tokens[c1]) + abs_v)
                # c1_pre in self.bi_tokens:
                else:
                    nominator = (self.bi_tokens[c1_pre]+1)/(self.uni_tokens[pre_word] + abs_v)
                    if c1_next not in self.bi_tokens:
                        p_c1 = nominator * (1/((self.uni_tokens[c1]) + abs_v))
                    else:
                        p_c1 = nominator * (self.bi_tokens[c1_next]+1)/((self.uni_tokens[c1]) + abs_v)
                ## c2 : candidate2
                # c2_pre not in self.bi_tokens:
                if c2_pre not in self.bi_tokens:
                    if c2_next not in self.bi_tokens:
                        p_c2 = (1/abs_v) * (1/abs_v)
                    else:
                        p_c2 = (1/abs_v) * (self.bi_tokens[c2_next] + 1)/((self.uni_tokens[c2]) + abs_v)
                # c2_pre in self.bi_tokens:
                else:
                    nominator = (self.bi_tokens[c2_pre]+1)/(self.uni_tokens[pre_word] + abs_v)
                    if c2_next not in self.bi_tokens:
                        p_c2 = nominator * (1/((self.uni_tokens[c2]) + abs_v))
                    else:
                        p_c2 = nominator * (self.bi_tokens[c2_next]+1)/((self.uni_tokens[c2]) + abs_v)
            # bigram without smoothing
            # P(x(n)|x(n-1)) = counts(x(n-1),x(n)) / counts(x(n-1))
            else:
                if c1_pre in self.bi_tokens and c1_next in self.bi_tokens:
                    p_c1 = (self.bi_tokens[c1_pre]/self.uni_tokens[pre_word]) * (self.bi_tokens[c1_next]/self.uni_tokens[c1])
                if c2_pre in self.bi_tokens and c2_next in self.bi_tokens: 
                    p_c2 = (self.bi_tokens[c2_pre]/self.uni_tokens[pre_word]) * (self.bi_tokens[c2_next]/self.uni_tokens[c2])     
            # append the candidate with the bigger probabilty in the result list
            if p_c1 < p_c2 : result.append((c2, p_c2))
            elif p_c1 == p_c2 : 
                if p_c1 == 0: 
                    result.append(("ZERO probability", 0))
                else: result.append(("SAME probability", 0))
            else : result.append((c1, p_c1))
        # return the list of picked candidates
        return result
    def get_pre_n_next(self, questions):
        # save pre,next words of '____'
        pre_next = list()
        for q in questions:
            # index of '____'
            q = q.split()
            for i in range(len(q)): 
                if q[i] == '____': 
                    pre_next.append([q[i-1], q[i+1]])
                    break
        return pre_next

def read_questions(path):
    lines = open(path, 'r').readlines()
    questions = list()
    candidates = list()
    for l in lines:
        q, c = l.split(':')
        questions.append(q.strip())
        candidates.append(c.strip().split('/'))
    return questions, candidates

def print_answers(questions, result):
    for i in range(len(questions)):
        if result[i][1] != 0:
            print(questions[i].replace("____", str(result[i][0])))
        else:
            print(result[i][0])

if __name__  == '__main__':
    # load corpus and questions
    path_c = os.path.join( os.getcwd(), sys.argv[1] )
    path_q = os.path.join( os.getcwd(), sys.argv[2] )
    # 500,000 training sentences
    corpus = open(path_c, 'r').readlines()
    lm = LM(corpus)
    
    # load 10 questions and each candidates
    questions, candidates = read_questions(path_q)
    # 'unigram - language model1'
    result_model1 = lm.get_prob_uni(questions, candidates)
    print('\n[ uni-gram ] \n')
    print_answers(questions, result_model1)
    # --------------------------------------------------------------------------
    # 'bigram without smoothing - language model2'
    result_model2 = lm.get_prob_bi(questions, candidates)
    print('\n[ bi-gram ] \n')
    print_answers(questions, result_model2)
    # --------------------------------------------------------------------------
    # 'bigram with smoothing - language model3'
    result_model3 = lm.get_prob_bi(questions, candidates, True)
    print('\n[ bi-gram with smoothing ] \n')
    print_answers(questions, result_model3)
