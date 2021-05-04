# Assignment 3
### Submitted by :
    1) Ayush Sanghavi (sanghavi)
    2) Vighnesh Kolhatkar (vkolhatk)
    3) Vishwas Desai (visdesai)


# Part 1: Part-of-speech tagging

Part of Speech Tagging (POS) is a process of tagging sentences with part of speech such as nouns, verbs, adjectives and adverbs, etc.


### Problem Statement
Use the training corpus to estimate parameters and, display the output a) Simple Model b) HMM c) Complex model.
The problem consists of 3 parts:-

    - Performing a Parts of Speech tagging on a simplified Bayes Net
    - Performing Parts of Speech tagging using HMM / Viterbi algorithm
    - Performing Parts of Speech tagging by implementing Gibbs Sampling


--------
Firstly, by using the trained data, we store the emission, transition probabilities in a dictionary.

### Bayes Net
The bayes net probability was calculated and thenn stored in a dictionary. The probability of each word given speech was stored and normalized.

### Formulating the estimation:

Consider W as the input word sequence and T is an input tag (POS) sequence i.e.

`W = w1,w2,…wn`

`T = t1,t2,…tn`

Our goal is to find a tag sequence T that maximizes P(T | W).

Using Bayes rule:

`maximize P(T∣W)=P(W∣T)∗P(T)/P(W)∝P(W∣T)∗P(T)`

----------
## Implementation
### Getting the posterior probability

- Normalize the sum of all the words and pos from their dictionaries.
  
- For Bayes Net, we calculate the posterior probability using the emission probability.
  
- For the bayes net we calculate the probability using the emission probability. 
  
- Similiarly for the Viterbi algorithm, we make us of the emission probability and transition probabilty in getting the posterior probability. 

- And for Gibbs sampling, we calculate the probability using the emission, transions probabilty and Parts of Speech count.
Emission Probability

----------
#### Emission Probability : Each word's emission probability was calculated and stored in a dictionary. Emission probability is a set of probabilities that each word is of the correct parts of speech i.e. the probabilty of the word 'Bruce' being a noun etc.

---------
#### Transmission Probability: The likelihood of a particular sequence, such as a noun being preceded by a model, a model by a verb, and a verb by a noun, is known as the transition probability. The probability should be high for a sequence to be right.

---------
#### Viterbi: The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events. After applying the Viterbi algorithm the model tags the sentence into its correct tags.

---------
#### Gibbs sampling: The ability to sample from the target distribution's conditional distributions. Given a joint distribution and sampling them iteratively by its conditional distributions will give a samples which contain tags that will occur frquenlty of a particular word which is then assigned as its final tag.
-------
### References: 
- https://www.mygreatlearning.com/blog/pos-tagging/
- https://courses.engr.illinois.edu/cs440/fa2018/Lectures/lect20.html
- https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/
- https://medium.com/analytics-vidhya/parts-of-speech-pos-and-viterbi-algorithm-3a5d54dfb346





# Part 2: 
# Mountain finding using Simple Bayes' Net, Viterbi and Human feedback
## Problem Statement : 
- Identifying the shapes of the mountain

- We are trying to find the ridge line between the mountain and the sky using Simple Bayes' Net, Viterbi and Human feedback

- Estimating the corresponding column for each ridge line

- We pass the following System arguments;
  1) The sample image
  2) Row co-ordinate and Column co-ordinate
  
## Motivation :

- This technique is useful for army, flight travelling, and other useful and safety applications.
- This kind of image processing helps us automate a lot of things.
- We compare simple bayes' net, viterbi algorithm and human feedback outputs here.

## Implementation : 

### edge_strength
-We have been given a function namely edge_strength that measures how strong the image gradient is at each point.

### draw_edge
- This function helps us draw the edge on the image with the correct amount of thickness and color.

------------

- For detecting the ridge using Simple Bayes' net we find the maximum edge strength per column and return an array.  The denominator can be ignored for all the terms.
- For detecting the ridge using Viterbi algorithm, we use defined transition probabilities, sum the edge strength and define two arrays
- One is the final array and the other one is to store the previous state for back tracking.
- We then calculate the initial state probabilities.
- We then calculate the state probabilities of each node.
- We go back to the previous solution that was memoized and then output the image.
- For human feedback, we take the human input values as our state probabilities.
- We propagate through the change, and maximize to find the solution.

---------
- For ridge detection using Simple Bayes' nets, we use the color : Red
- For ridge detection using Viterbi, we use the color : Blue
- For ridge detection using human feedback, we use the color : Green
----------
Sample Test Image example: 

- Output

![mountain](../master/part2/output_images/mountain_output.jpg)


## References: 

- https://www.cse.unr.edu/~bebis/ISVC13_Horizon.pdf
- https://link.springer.com/article/10.1007/s00773-017-0464-8




# Part 3:
# Reading text using Simple Bayes' Net, Variable Elimination and Viterbi Algorithm
## Problem Statement : 
- Our goal is to recognize the text in the images given

- But the images are noisy.

- Assumption : All text has same font and font size. We’ll also assume that our text only has the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, space, and 7 punctuation symbols.

- We pass the following system arguments ; 
  1) courier-train.png that has the image of the letters used for training.
  2) The train-text.txt; that is the representative of English Language.
  3) Passing one of test image.
    
## Motivation : 
- Using the advantages of Simple Bayes' nets and HMM we are trying to actually implement and recognize the text.
- Most modern systems in this technological era are good at recognizing texts but it gets difficult when the character is weird or lack of training data or when the character is isolated.
- Using the concept of HMM, viterbi algorithm and probabilities, we can incorporate the art of recognizing the characters of the texts.  


## Implementation: 

### load_letters : 
- This function was given to us as a starter code.
- It makes sure about the font size and represents the image as List of Lists in which the characters are represented as a 2D grid of black and white dots.

### load_training_letters:
- Loading the 26 alphabets, in both lower and upper cases, 10 digits, space, and 7 punctuation symbols.

-------
- We are then calculating the transition probabilities of each Character.
- Transition probability is the probability of moving from one state to another.
- We then create a char_fringe which stores the characters as list.
- We calculate the initial probabilities and then the emission probabilities.
- After we have initial probabilities, transition probabilities and emission probabilities we implement different methods to
  recognize the text in an image provided as test-image file and display the recognized text.
-------
- First Approach : Our first approach was Simple Bayes' Net.
- Second Approach : HMM Variable Elimination method
- Third Approach : Viterbi Algorithm
-------
- We are using a sample text file as a texual training data file to calculate the initial probabilities as well as
  the transition probabilities. 
- To calculate the emission probability; We compare the pixels and the pixel info of each of the character of the image training data.
- For the same; we have used the pixel information we have. There would be two cases: first is when the image is densely populated with black pixels and second is when the image has sparse black pixels.
- We assign some weights to black, less black pixels and non matching pixels where the test image character's pixel is white and train image character's pixel is black.
-------
- For Variable elimination:
- To perform var elimination we have called two functions : calc_fwd_prob and calc_bwd_prob on chars of the test image.
  We marginalize the near-by chars to estimate the target char.  
-------
- For viterbi algorithm ; state1 is our current state and state0 is our previous state.
  We calculate the most probable state at time t, and find the previous hidden state.
- We use heapq (priority queue) to pop the states and find the previous max state.
------
- References: 
  
  - https://www.youtube.com/watch?v=kqSzLo9fenk
  
  - https://beginnersbook.com/2019/03/python-ord-function/#:~:text=The%20ord()%20function%20in,value%20of%20character%20'B'
  
  - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.6415&rep=rep1&type=pdf
  
  - https://towardsdatascience.com/optical-character-recognition-ocr-with-less-than-12-lines-of-code-using-python-48404218cccb
  
------



