# !/usr/local/bin/python3
#
# Authors: Ayush Sanghavi (sanghavi) -  Vighnesh Kolhatkar (vkolhatk)  -  Vishwas Desai (visdesai)
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
from numpy import zeros, array, sqrt


# calculate "Edge strength map" of an image

def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return sqrt(filtered_y ** 2)


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)
    return image


# main program

(input_filename, gt_row, gt_col) = sys.argv[1:]
gt_col = int(gt_col)
gt_row = int(gt_row)
# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]




# ---------# ---------# --------- Initializing values # ---------# ---------# ---------# ---------
column_length = edge_strength.shape[1]
row_length = edge_strength.shape[0]

# transition probability values
# for the state in same row or above or below or two rows above or below = 0.5

# for the rest, we assume transition probabilities to be zero and never calculate those transitions in viterbi
transition_probability = [0.5, 0.5, 0.5]

# final array
final = zeros((row_length, column_length))

# array which stores the previous state for back tracking
max_state = zeros((row_length, column_length))

viterbi_ridge = zeros(column_length)
slope = zeros(column_length)


# ---------# ---------# ---------# ---------  2.1 Simple Bayes Net  # ---------# ---------# ---------# ---------

def simple_bayes_net():
    bayes_ridge = argmax(edge_strength, axis=0)
    
    return bayes_ridge


# ---------# ---------# ---------# --------- 2.2  Viterbi    --------# ---------# ---------# ---------# ---------

def hmm_viterbi():
    # summing up the edge strength per column
    for col in range(column_length):
        for row in range(row_length):
            slope[col] += edge_strength[row][col]

    # initial state probabilities
    for row in range(row_length):
        final[row][0] = edge_strength[row][col] / slope[0]

    # state probabilities using Viterbi
    for col in range(1, column_length):
        for row in range(row_length):
            maximize = 0
            for j in range(-2, 3):
                if ((row + j < row_length) & (row + j >= 0)):
                    if (maximize < final[row + j][col - 1] * (transition_probability[abs(j)])):
                        maximize = (final[row + j][col - 1]) * (transition_probability[abs(j)])
                        max_state[row][col] = row + j
                    final[row][col] = (edge_strength[row][col] / 100) * (maximize)
                    # print(final[row][col])

    # node with maximize probability
    maximize = argmax(final[:, column_length - 1])

    # backtracking to solution based on memory storage of previous max product stored in viterbi
    for col in range(column_length - 1, -1, -1):
        viterbi_ridge[col] = int(maximize)
        maximize = max_state[int(maximize)][col]
        # print(maximize)
    
   
    return viterbi_ridge


# ---------# ---------# ---------# ---------#  Human Feedback  ---------# ---------# ---------# ---------# ---------

def human_feedback():
    human_ridge = [row_length / 4] * column_length

    # Taking the state probabilities
    for row in range(row_length):
        final[row][gt_col] = 0
    final[gt_row][gt_col] = 1

    # previous columns
    for col in range(gt_col - 1, 0, -1):
        for row in range(row_length - 1, -1, -1):
            maximize = 0
            for j in range(-2, 3):
                if ((row + j < row_length) & (row + j >= 0)):
                    if (maximize < final[row + j][col + 1] * (transition_probability[abs(j)])):
                        maximize = (final[row + j][col + 1]) * (transition_probability[abs(j)])
                        max_state[row][col] = row + j
                    final[row][col] = (edge_strength[row][col] / 100) * (maximize)

    # for columns ahead
    for col in range(gt_col + 1, column_length):
        for row in range(0, row_length):
            maximize = 0
            for j in range(-2, 3):
                if ((row + j < row_length) & (row + j >= 0)):
                    if (maximize < final[row + j][col - 1] * (transition_probability[abs(j)])):
                        maximize = (final[row + j][col - 1]) * (transition_probability[abs(j)])
                        max_state[row][col] = row + j
                    final[row][col] = (edge_strength[row][col] / 100) * (maximize)

    # backtracking to find the maximized solution
    maximize = argmax(final[:, column_length - 1])
    for col in range(column_length - 1, -1, -1):
        human_ridge[col] = int(maximize)
        maximize = max_state[int(maximize)][col]
    
    
    return human_ridge




bayes_ridge = simple_bayes_net()
input_image = draw_edge(input_image, bayes_ridge, (255, 0, 0), 5)

# answer for part 2.2, does viterbi implmentation with bottom up and memoization
viterbi_ridge = hmm_viterbi()
input_image = draw_edge(input_image, viterbi_ridge, (0, 0, 255), 5)

# solution for part2.3 human input and viterbi
human_ridge = human_feedback()
imageio.imwrite("output.jpg", draw_edge(input_image, human_ridge, (0, 255, 0), 5))
