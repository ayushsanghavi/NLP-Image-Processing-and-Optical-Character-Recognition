###################################
# CS B551 Spring 2021, Assignment #3
#
# Submitted by: Ayush Sanghavi (sanghavi)
#               Vighnesh Kolhatkar (vkolhatk)
#               Vishwas Desai (visdesai)
#
# (Based on skeleton code by D. Crandall)
#

from collections import Counter
import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        # initial probability, pos count , word count
        self.initial_probability, self.partsOfSpeech_Count, self.word_count = {}, {}, {}
        # transition probability,  emission probability
        self.transPob_dict, self.emissionProb_dict = {}, {}
        # word, probability of word and POS in training dataset
        # POS --> Parts Of Speech
        self.word_dict, self.wordProb_dict, self.posProb_dict = {}, {}, {}

    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    def posterior(self, model, sentence, label):
        # Simple
        if model == "Simple":
            posterior_prob = 0.0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.emissionProb_dict:
                    e = self.emissionProb_dict[(sentence[i], label[i])]
                    posterior_prob += math.log(e)
                else:
                    posterior_prob += math.log(1e-24)
            return posterior_prob

        # HMM
        elif model == "HMM":
            posterior_prob = 0.0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.emissionProb_dict:
                    e = self.emissionProb_dict[(sentence[i], label[i])]
                    posterior_prob += math.log(e)
                else:
                    posterior_prob += math.log(1e-24)

            partsofspeech_prev = label[0]
            for i in range(1, len(label)):
                transP = self.transPob_dict[(partsofspeech_prev, label[i])]
                posterior_prob += math.log(transP)
                partsofspeech_prev = label[i]
            return posterior_prob

        # Complex
        elif model == "Complex":
            posterior_prob = 0.0

            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.emissionProb_dict:
                    c = self.partsOfSpeech_Count[label[i]] * self.emissionProb_dict[(sentence[i], label[i])]
                    posterior_prob += math.log(c)
                else:
                    posterior_prob += math.log(1e-24)

            partsofspeech_prev = label[0]
            for i in range(1, len(label)):
                transP = self.transPob_dict[(partsofspeech_prev, label[i])]
                posterior_prob += math.log(transP)
                partsofspeech_prev = label[i]
            return posterior_prob

        else:
            print("Unknown algo!")

    # ----------------------------------------------------------------------------------------------------------------
    # Do the training!

    # Training the data
    def train(self, data):
        self.word_dict = {'adj': [], 'adv': [], 'adp': [], 'conj': [], 'det': [], 'noun': [], 'num': [], 'pron': [],
                          'prt': [], 'verb': [], 'x': [], '.': []}

        self.initial_probability = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0,
                                    'prt': 0, 'verb': 0, 'x': 0, '.': 0}

        self.transPob_dict = {}

        self.partsOfSpeech_Count = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0,
                                    'prt': 0, 'verb': 0, 'x': 0, '.': 0}

        count = 0
        for element in data:
            # pos --> Parts of Speech
            for pos in element[1]:
                self.partsOfSpeech_Count[pos] += 1
                count += 1
            for word in element[0]:
                if word in self.word_count:
                    self.word_count[word] += 1
                else:
                    self.word_count[word] = 1

        for pos in self.partsOfSpeech_Count:
            self.partsOfSpeech_Count[pos] /= count
        for word in self.word_count:
            self.word_count[word] /= count
        for element in data:
            if len(element[1]) > 1:
                for i in range(len(element[1]) - 1):
                    pre = element[1][i]
                    nxt = element[1][i + 1]
                    self.initial_probability[pre] += 1
                    if (pre, nxt) not in self.transPob_dict:
                        self.transPob_dict.update({(pre, nxt): 1})
                    else:
                        self.transPob_dict[(pre, nxt)] += 1
                    self.word_dict[element[1][i]].append(element[0][i])
                self.word_dict[element[1][i + 1]].append(element[0][i + 1])

        norm = sum(self.initial_probability.values())

        for element in self.initial_probability.keys():
            self.initial_probability[element] /= norm

        for element in data:
            for i in range(len(element[0])):
                if element[0][i] not in self.posProb_dict:
                    self.posProb_dict[element[0][i]] = [element[1][i]]
                else:
                    self.posProb_dict[element[0][i]].append(element[1][i])

        sum_tp = sum(self.transPob_dict.values())
        for i in self.transPob_dict:
            self.transPob_dict[i] = (float(self.transPob_dict[i] / sum_tp))

        self.emissionProb_dict = {}

        for pos, words in self.word_dict.items():
            counts = Counter(words)
            counts = Counter(th for th in counts.elements())

        for pos, words in self.word_dict.items():
            norm = 0
            counts = Counter(words)
            counts = Counter(th for th in counts.elements())
            for element, c in counts.items():
                norm += c
            for element, c in counts.items():
                self.emissionProb_dict.update({(element, pos): c / norm})

        self.wordProb_dict = {}

        for word, pos in self.posProb_dict.items():
            norm = 0
            counts = Counter(pos)
            counts = Counter(j for j in counts.elements())
            for element, c in counts.items():
                norm += c
            for element, c in counts.items():
                if word not in self.wordProb_dict:
                    self.wordProb_dict.update({word: [[c / norm, element]]})
                else:
                    self.wordProb_dict[word].append([c / norm, element])

        x = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']

        # replacing all the empty elements in the transition probability
        # by a downscaled value.

        tmp = []
        for i in x:
            for j in x:
                tmp.append((i, j))

        tp_list = list(self.transPob_dict)

        diff = list(set(tmp) - set(tp_list))
        min_tp = min(self.transPob_dict.values())
        for i in diff:
            self.transPob_dict[i] = min_tp / 20.0
        pass

    # Functions for each algorithm. Right now this just returns nouns -- fix this!

    # Simplified Bayes Net
    def simplified(self, sentence):
        res = []
        for word in list(sentence):
            if word in self.posProb_dict:
                tmp = []
                for element in self.wordProb_dict[word]:
                    tmp.append(element[0])
                res.append(self.wordProb_dict[word][tmp.index(max(tmp))][1])
            else:
                res.append('x')
        return res

    # HMM Viterbi model
    def hmm_viterbi(self, sentence):
        curr = {'adj': [], 'adv': [], 'adp': [], 'conj': [], 'det': [], 'noun': [], 'num': [], 'pron': [], 'prt': [],
                'verb': [], 'x': [], '.': []}

        pos_st = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']

        res = []
        pos_pre = ''

        speech = {'adj': None, 'adv': None, 'adp': None, 'conj': None, 'det': None, 'noun': None, 'num': None,
                  'pron': None, 'prt': None, 'verb': None, 'x': None, '.': None}

        word = list(sentence)[0]
        e = 0.0
        for pos in pos_st:
            if (word, pos) in self.emissionProb_dict:
                e = self.emissionProb_dict[(word, pos)] * self.initial_probability[pos]
                curr[pos].append(("noun", e))
            else:
                curr[pos].append(("noun", 1e-8))

        res.append(pos_pre)

        for word in list(sentence)[1:]:
            t = speech
            for pos in pos_st:
                max_p = 0.0
                for pos_pre, prev_prob in curr.items():
                    if (word, pos) in self.emissionProb_dict:
                        tmp = prev_prob[-1][1] * self.emissionProb_dict[(word, pos)] * self.transPob_dict[
                            (pos_pre, pos)]
                        if max_p < tmp:
                            max_p = tmp
                            new_pos = pos_pre
                if max_p == 0.0:
                    for pos_pre, prev_prob in curr.items():
                        tmp = prev_prob[-1][1] * self.transPob_dict[(pos_pre, pos)] * 5e-7
                        if max_p < tmp:
                            max_p = tmp
                            new_pos = pos_pre
                t[pos] = (new_pos, max_p)

            for i, j in t.items():
                curr[i].append(j)

        max_p = 0.0
        pos_t = ""
        tmp = ""

        for pos, word in curr.items():
            if max_p < word[-1][1]:
                max_p = word[-1][1]
                pos_t = word[-1][0]
                tmp = pos

        max_p = 0.0
        res = []

        if not tmp:
            tmp = 'noun'

        res.append(tmp)
        if not pos_t:
            max_p = 0.0
            for pos in pos_st:
                if max_p < self.transPob_dict[(res[-1], pos)]:
                    max_p = self.transPob_dict[(res[-1], pos)]
                    pos_t = pos

        res.append(pos_t)

        for i in range(len(sentence) - 2, 0, -1):
            if pos_t:
                pos_t = curr[pos_t][i][0]
            else:
                max_p = 0.0
                for pos in pos_st:
                    if max_p < self.transPob_dict[res[-1]][pos]:
                        max_p = self.transPob_dict[res[-1]][pos]
                        pos_t = pos
            res.append(pos_t)
        res.reverse()

        return res[:len(sentence)]

    # Complex
    def complex_mcmc(self, sentence):
        res = ["noun"] * len(sentence)
        states = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        bias_p = []

        for word in list(sentence):
            tmp = []
            for pos in states:
                if (word, pos) in self.emissionProb_dict:
                    tmp.append(self.emissionProb_dict[(word, pos)])
                else:
                    tmp.append(2.0)

            min_tmp = min(tmp)

            for i in range(len(tmp)):
                if tmp[i] == 2.0:
                    tmp[i] = min_tmp * 1e-12
            sum_tmp = sum(tmp)

            for i in range(len(tmp)):
                tmp[i] /= sum_tmp
            x = 0.0
            for i in range(len(tmp)):
                x += tmp[i]
                tmp[i] = x
            bias_p.append(tmp)

        res = []
        s = 0
        for s in range(len(sentence)):
            l = 1000
            res.append([])
            while l > 0:
                l -= 1
                rand = random.random()
                for i in range(12):
                    if rand <= bias_p[s][i]:
                        res[s].append(states[i])
                        break

        soln = []
        for element in res:
            soln.append(element[-1])
        return soln

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labellings of the sentence, one
    #  part of speech per word.

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

# ------------------------------------------- References ------------------------------------------------

# https://www.mygreatlearning.com/blog/pos-tagging/
# https://courses.engr.illinois.edu/cs440/fa2018/Lectures/lect20.html
# https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/
