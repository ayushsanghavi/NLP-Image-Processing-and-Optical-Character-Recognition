#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Ayush Sanghavi (sanghavi) - Vighnesh Kolhatkar (vkolhatk) - Vishwas Desai (visdesai)
# (based on skeleton code by D. Crandall, Oct 2020)


# Importing all the necessary libraries
# https://pypi.org/project/Pillow/
from PIL import Image, ImageDraw, ImageFont
import sys
import math
import copy
import heapq
import re
# from scipy import special
# import pandas

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

class Char_recognition:
    def __init__(self):
        self.initial_probability = {}
        self.transition_probability = {}
        self.emission_probability = {}


    # Loading the letters
    def load_letters(self, fname):
        im = Image.open(fname)
        px = im.load()
        (x_size, y_size) = im.size
        result = []
        for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
            result += [['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) for y in
                        range(0, CHARACTER_HEIGHT)], ]
        return result


    # Loading the training letters -
    def load_training_letters(self, fname):
        TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        letter_images = self.load_letters(fname)
        return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

    # Here, we are calculating the transition probability.

    def calculate_transition_probability(self):
        for char in self.transition_probability:
            sum_of_frequency = sum(self.transition_probability[char][letter] for letter in self.transition_probability[char])
            for letter in self.transition_probability[char]:
                self.transition_probability[char][letter] = self.transition_probability[char][letter] / float(sum_of_frequency)

    # Calculate initial probability and transition probability
    def calculate_transition_probability_initial_probability(self, train_txt_fname):
        f = open(train_txt_fname, 'r'); # Reading the train text file.
        line_n = 0
        for line in f:
            line_n += 1
            char_fringe = list(re.sub(r'[&|$|*|;|`|#|@|%|^|~|/|<|>|:|[|\]|{|}|+|=|_]', r'', " ".join([word for word in line.split()][0::2])))
            if char_fringe:
                self.initial_probability[char_fringe[0]] = self.initial_probability.get(char_fringe[0], 0) + 1
                for letter in range(1, len(char_fringe)):
                    if char_fringe[letter - 1] in self.transition_probability:
                        self.transition_probability[char_fringe[letter - 1]][char_fringe[letter]] = self.transition_probability[char_fringe[letter - 1]].get(char_fringe[letter], 0) + 1
                    else:
                        temp = {char_fringe[letter]: 1}
                        self.transition_probability[char_fringe[letter - 1]] = temp

        sum_initial_probability = sum(self.initial_probability[letter] for letter in self.initial_probability)

        for x in self.initial_probability:
            self.initial_probability[x] = float(self.initial_probability[x]) /float(sum_initial_probability)
        self.calculate_transition_probability()

    # First approach : Simple Bayes' net
    # Estimating the high probable character in the image using the emission probability of the character seen at it's positon.
    def simple_bayes_net(self):
        simple = ""
        for char in self.emission_probability:
            simple += "".join(max(self.emission_probability[char], key=lambda x: self.emission_probability[char][x]))
        print(" Simple: {0}".format(simple))

    # Calculating the emission probability using the scenario when the test image is densely black pixelated and when the image is sparsely black pixelated.
    def calculate_emission_probability(self):
        black_test_letter = 0
        black_train_letter = 0
        for letter in test_letters:
            for i in letter:
                if i == '*':
                    black_test_letter += 1
        for letter in train_letters:
            for i in train_letters[letter]:
                if i == '*':
                    black_train_letter += 1
        for test_letter in range(len(test_letters)):
            self.emission_probability[test_letter] = {}
            for train_letter in train_letters:
                black_count = 0
                white_count = 0
                black_non_matching = 0
                white_non_matching = 0
                total = 0
                for char in range(len(test_letters[test_letter])):
                    total += 1
                    if test_letters[test_letter][char] == train_letters[train_letter][char] and \
                                    train_letters[train_letter][char] == '*':
                        black_count += 1
                    elif test_letters[test_letter][char] == train_letters[train_letter][char] and \
                                    train_letters[train_letter][char] == ' ':
                        white_count += 1
                    elif train_letters[train_letter][char] == '*':
                        black_non_matching += 1
                    elif train_letters[train_letter][char] == ' ':
                        white_non_matching += 1

                if black_test_letter / len(test_letters) > black_train_letter / len(train_letters):
                    self.emission_probability[test_letter][train_letter] = math.pow(0.8, black_count) * math.pow(0.7,white_count) * math.pow(0.3, black_non_matching) * \
                                                                    math.pow(0.2, white_non_matching)
                else:
                    self.emission_probability[test_letter][train_letter] = math.pow(0.99, black_count) * math.pow(0.7,white_count) * math.pow(0.3, black_non_matching) * \
                                                                    math.pow(0.01, white_non_matching)

    # Second approach : HMM using variable elimination method
    # To perform var elimination we have called two functions : calc_fwd_prob and calc_bwd_prob on chars of the test image.
    # We marginalize the near-by chars to estimate the target char.
    def hmm_variable_elimination(self):
        # forward algorithm
        forward = self.calc_prob_fwd()
        # calc_prob_bwd
        backward = self.calc_prob_bwd()
        # find posterior seq max for each state
        seq = []
        letter_len = len(test_letters) - 1
        for i in range(len(test_letters)):
            dictionary_probability = {category: backward[letter_len][category] * forward[i][category] for category in train_letters}
            seq.append(max(dictionary_probability, key=dictionary_probability.get))
            letter_len -= 1
        print(" HMM VE: {0}".format(''.join(seq)))
        
   

    # Calculate probability using forward algorithm
    def calc_prob_fwd(self):
        forward = []
        forward_previous = {}
        for i, word in enumerate(test_letters):
            forward_current = {}
            for j, category in enumerate(train_letters):
                if i == 0:
                    sum_pred = self.initial_probability.get(category, math.pow(10, -6))
                else:
                    sum_pred = sum(forward_previous[ltr] * self.transition_probability.get(ltr, {}).get(category, math.pow(10, -6)) for ltr in train_letters)
                forward_current[category] = sum_pred * self.emission_probability[i].get(category, math.pow(10, -6))
            forward_current = {key: value / max(forward_current.values()) for key, value in forward_current.items()}
            forward.append(forward_current)
            forward_previous = forward_current
        return forward

    # Calculate probability using backward algorithm
    def calc_prob_bwd(self):
        backward = []
        backward_previous = {}
        for i, word in enumerate(test_letters[::-1]):
            backward_current = {}
            for j, category in enumerate(train_letters):
                if i == 0:
                    backward_current[category] = 1
                else:
                    word_succ = i - 1
                    backward_current[category] = sum(
                        backward_previous[next_state] * self.emission_probability[word_succ].get(next_state, math.pow(10, -6)) *
                        self.transition_probability.get(category, {}).get(next_state, math.pow(10, -6)) for k, next_state in
                        enumerate(train_letters))
            backward_current = {key: value / max(backward_current.values()) for key, value in backward_current.items()}
            backward.append(backward_current)
            backward_previous = backward_current
        return backward


    # Third Approach : Viterbi algorithm
    def hmm_MAP(self):
        state1 = [None] * 128
        state0 = [None] * 128
        for states, state in enumerate(test_letters):
            for index, current_char in enumerate(train_letters):
                if states == 0:
                    result = -math.log(self.emission_probability[0][current_char]) - math.log(self.initial_probability.get(current_char, math.pow(10, -8)))
                    state1[ord(current_char)] = [result, [current_char]]
                else:
                    min = []
                    for prev_char_index, previous_char in enumerate(train_letters):
                        prev_prob = -math.log(self.transition_probability.get(previous_char, {}).get(current_char, math.pow(10, -8))) + state0[ord(previous_char)][0]
                        min.append([prev_prob, state0[ord(previous_char)][1] + [current_char]])
                    heapq.heapify(min)
                    prev_transition_max = heapq.heappop(min)
                    result = prev_transition_max[0] - math.log(self.emission_probability[states][current_char])
                    state1[ord(current_char)] = [result, prev_transition_max[1]]
            state0 = copy.deepcopy(state1)
            state1 = [None] * 128
        final = []
        for element in state0:
            if element is not None:
                final.append(element)
        heapq.heapify(final)
        result = heapq.heappop(final)
        print("HMM MAP: {0}".format(''.join(result[1])))





#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

#Instantiating the class
Char_recognition = Char_recognition()
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = Char_recognition.load_training_letters(train_img_fname)
test_letters = Char_recognition.load_letters(test_img_fname)

# calling all the functions using object name
Char_recognition.calculate_transition_probability_initial_probability(train_txt_fname)
Char_recognition.calculate_emission_probability()
Char_recognition.simple_bayes_net()
Char_recognition.hmm_variable_elimination()
Char_recognition.hmm_MAP()



## Below is just some sample code to show you how the functions above work.
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:

#print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:

#print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:

# print("Simple: " + "Sample s1mple resu1t")
# print("   HMM: " + "Sample simple result")


