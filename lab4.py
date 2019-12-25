#180228768

from collections import Counter
import sys
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pprint import pprint

random.seed(11242)
depochs = 5
feat_red = 0
print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)
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

train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])

all_tags = ["O", "PER", "LOC", "ORG", "MISC"]
# feature space of cw_ct
def cw_ct_counts(data, freq_thresh = 5): # data inputted as (cur_word, cur_tag)
    cw_c1_c = Counter()
    for doc in data:
        cw_c1_c.update(Counter(doc))
    return Counter({k:v for k,v in cw_c1_c.items() if v > freq_thresh})
cw_ct_count = cw_ct_counts(train_data, freq_thresh = feat_red)
# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts): # sent as (cur_word, cur_tag)
    phi_1 = Counter()
    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_count.keys()])
    return phi_1
sent = train_data[0]
phi_1(sent, cw_ct_count)
# feature space of pt-ct
def pt_ct_counts(data, freq_thresh = 5): # input (cur_word, cur_tag) 
    tagtag = Counter()
    for doc in data:
        tags = list(zip(*doc))[1]
        for i in range(len(tags)):
            if i == 0:
                tagtag.update([("*", tags[i])])
            else:
                tagtag.update([(tags[i-1], tags[i])])
    # return feature space with features with counts above freq_thresh
    return Counter({k:v for k,v in tagtag.items() if v > freq_thresh})
pt_ct_count = pt_ct_counts(train_data, freq_thresh = feat_red)
# combining feature spaces
comb_featspaces = pt_ct_count + cw_ct_count
# creating our sentence features
def phi_2(sent, pt_ct_count):
    sentence, tags = zip(*sent)
    tags = ["*"] + list(tags)
    # returning features if found in the feature space
    tags = [(tags[i], tags[i+1]) for i in range(len(tags)-1) if (tags[i], tags[i+1]) in pt_ct_count]
    return Counter(tags)
sent = train_data[0]    
phi_2(sent, pt_ct_count)

"""Perceprton"""
class Perceptron():
    def __init__(self,all_tags):
        super(Perceptron, self).__init__()
        self.all_tags = all_tags
    
    # creating all possible combinaions of 
    def pos_combos(self,sentence):
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags,repeat=len(sentence))]
        return combos
    
    def scoring(self,doc, weights, extra_feat = True):
        # unzippin them
        sentence, tags = list(zip(*doc))
        # all possible combos of sequences
        combos = list(enumerate(self.pos_combos(sentence)))
        # our score matrix
        scores = np.zeros(len(combos))
        # looping through all possible combos
        for index, sent_tag in combos:
            if extra_feat is False:
                # retrieving the counter if its in our feature space
                phi = phi_1(sent_tag, cw_ct_count)
            else:
                phi1 = phi_1(sent_tag, cw_ct_count)
                phi2 = phi_2(sent_tag, pt_ct_count)
                phi = phi1 + phi2
            # if its not then the score is 0
            if len(phi) == 0:
                    scores[index] = 0
            else:
                temp_score = 0
                # otherwise do the w*local_phi
                for pair in phi:
                    if pair in weights:
                        print('pair  : ', pair)
                        temp_score += weights[pair]*phi[pair]
                        return
                    else:
                        temp_score += 0
                # store the score with the index
                scores[index] = temp_score
        # retrieve the index of the highest scoring sequence
        max_scoring_position = np.argmax(scores)
        # retrieve the highest scoring sequence
        max_scoring_seq = combos[max_scoring_position][1]
        return max_scoring_seq
    
    def viterbi(self, doc, weights):
        words, _ = list(zip(*doc))
        nwords = len(words)
        ntags = len(all_tags)
        V = np.zeros((nwords, ntags))
        B = np.zeros((nwords, ntags))

        indice = list()
        max_scoring_seq = list()
        final = 0
        
        for i, word in enumerate(words):
            poss = [(word, tag) for tag in all_tags]
            phi = phi_1(poss, cw_ct_count)
            for j in range(ntags):
                pair = poss[j]                
                if i == 0 and pair in weights:
                    V[i][j] = weights[pair] * phi[pair]
                else :
                    B[i][j] = np.argmax(V[i-1])
                    if pair in weights:
                        V[i][j] = np.max(V[i-1]) + weights[pair] * phi[pair]
            if i == nwords-1 :
                final = np.argmax(V[i])
        indice.append(final)
        
        ### Backtrace with Backpointer matrix(B) for prediction
        for i in range(nwords-1, 0, -1):
            indice.insert(0, int(max(B[i])))
        for i, j in enumerate(indice): 
            max_scoring_seq.append((words[i], all_tags[j]))
        return max_scoring_seq

    def beam(self, doc, weights):
        k = 3 
        words, _ = list(zip(*doc))
        ntags = len(all_tags)
        scores = [] 
        B = []
        for i, word in enumerate(words):
            poss = [(word, tag) for tag in all_tags]
            phi = phi_1(poss, cw_ct_count)
            if i == 0:    
                for j in range(ntags):
                    pair = poss[j]                
                    scores.append(([pair], weights[pair] * phi[pair]))
                B = sorted(scores, key = lambda tup : tup[1], reverse = True)[:k]
            else: 
                b = []
                for seq, score in B:
                    for j in range(ntags):
                        pair = poss[j]
                        b.append(((seq + [pair]), weights[pair] * phi[pair] + score))
                B = sorted(b, key = lambda tup : tup[1], reverse = True)[:k]
        top1 = B[0]
        max_scoring_seq = top1[0]
        return max_scoring_seq

    def train_perceptron(self, data, epochs, shuffle = True, extra_feat = False):
        # variables used as metrics for performance and accuracy
        iterations = range(len(data)*epochs)
        false_prediction = 0
        false_predictions = []
        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()
        start = time.time()
        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()
            # going through each sentence-tag_seq pair in training_data
            # shuffling if necessary
            if shuffle == True:
                random.shuffle(data)
            for doc in data:
                # retrieve the highest scoring sequence
                if(sys.argv[1] == '-b') : 
                    max_scoring_seq = self.beam(doc, weights)
                elif(sys.argv[1] == '-v') : 
                    max_scoring_seq = self.viterbi(doc, weights)
                else : 
                    max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)
                # if the prediction is wrong
                if max_scoring_seq != doc:
                    correct = Counter(doc)
                    # negate the sign of predicted wrong
                    predicted = Counter({k:-v for k,v in Counter(max_scoring_seq).items()})
                    # add correct
                    weights.update(correct)
                    # negate false
                    weights.update(predicted)
                    
                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)
            print("Epoch: ", epoch+1, 
                  " / Time for epoch: ", round(time.time() - now,2),
                 " / No. of false predictions: ", false)
        return weights, false_predictions, iterations
    
    # testing the learned weights
    def test_perceptron(self,data, weights, extra_feat = False):
        correct_tags = []
        predicted_tags = []
#        i = 0
        for doc in data:
            _, tags = list(zip(*doc))
            correct_tags.extend(tags)
            if(sys.argv[1] == '-b') : 
                max_scoring_seq = self.beam(doc, weights)
            elif(sys.argv[1] == '-v') : 
                max_scoring_seq = self.viterbi(doc, weights)
            else :  
                max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)
            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tags.extend(pred_tags)
        return correct_tags, predicted_tags
    
    def evaluate(self,correct_tags, predicted_tags):
        all_tags = ["PER", "LOC", "ORG", "MISC"]
        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=all_tags)
        print("F1 Score: ", round(f1, 5))
        return f1
        
perceptron = Perceptron(all_tags)
print("\nTraining the perceptron with (cur_word, cur_tag) \n")
weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs = depochs)
print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")
correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)
f1 = perceptron.evaluate(correct_tags, predicted_tags)