'''
Ben Blease
4/27/17
Hidden Markov Model implementation
Input seq.txt
'''

import sys
import math
import random
import numpy as np
import copy
import sys

'''
Calculate the posterior probabilities
'''
def forward_backward(inp, A, B, init):
    #forward probabilities
    forward = np.zeros((1000, 3))
    alpha = (init * B[inp[0] - 1])[0]
    forward[0] = alpha
    for i in range(1, len(inp)):
        k = []
        for j in range(3):
            k += [np.dot(A[j], alpha) * B[inp[i] - 1][j]]


        alpha = k/np.sum(k)
        forward[i] = alpha

    #backward probabilities
    backward = np.zeros((1000, 3))
    beta = [1 for _ in range(3)]
    backward[-1] = beta
    for i in range(len(inp) - 2, 0 - 1, -1):
        k = []
        for j in range(3):
            k += [np.sum(A[j]* beta * B[inp[i + 1] - 1][j])]

        beta = k/np.sum(k)
        backward[i] = beta
    #normalize and return a 3 x 1000 matrix
    out = forward * backward
    out[out == 0] = 1e-100
    out = out / np.outer(np.sum(out, axis=1), [1, 1, 1])
    return out

'''
Re estimate the Transition and Output matrices
'''
def MLEA_B(inp, z):
    #count all transitions from one state to the next
    At = np.zeros((3, 3))
    Ab = np.zeros((1, 3))
    for i in range(1, len(z)):
        At[z[i - 1]][z[i]] += 1
    Ab = np.outer(np.sum(At, axis=1), [1, 1, 1])
    Anew = At/Ab

    #count all outputs and states
    Bt = np.zeros((10, 3))
    Bb = np.zeros((1, 3))
    for i in range(1, len(z)):
        Bt[inp[i] - 1][z[i]] += 1
    Bb = np.sum(Bt, axis=0)
    Bnew = Bt/np.outer(Bb, [1 for _ in range(10)]).transpose()

    return (Bnew, Anew)

def read_seq():
    f = open("seq.txt", 'r')
    seq = f.read().split(" ")
    return list(map(int, seq))


def main(argc, argv):
    if (argc != 3):
        print("Usage : python3 hw3.py <fixed A> <iterations>")
        return

    seq = read_seq()
    fixed = True if argv[1] == "true" or argv[1] == "True" else False
    iterations = int(argv[2])

    #if A is fixed, set to correct answer and don't update
    if (fixed):
        A = np.array([[0.75, 0.25, 0.0], [0.25, 0.5, 0.25], [0.0, 0.25, 0.75]])
    else:
        A = np.random.rand(3, 3)
        A = A / np.sum(A, axis=0)
        A = A.transpose()

    B = np.random.rand(10, 3)
    B = B / np.sum(B, axis=0)
    init = np.random.rand(3)
    init = init / np.outer(np.sum(init), [1, 1, 1])
    c = 0
    while(True):
        oB = B
        oA = A
        pos = forward_backward(seq, copy.deepcopy(A), copy.deepcopy(B), init)
        init = [pos[0]]
        #sample from the posterior using a random choice and the posterior distribution
        z = [np.random.choice(np.arange(0, 3), p=pos[i]) for i in range(len(pos))]
        res = MLEA_B(seq, z)
        B = res[0]
        if not fixed:
            A = res[1]
        c += 1
        if c == iterations:
             break
    print(A)
    print(B)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
