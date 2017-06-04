'''
Ben Blease
2/17/17
I pledge my honor that I have abided by the Stevens honor system.
Logistic regression for NESARC data predicting variable 313
'''
import csv
import math
import sys
import numpy as np

sheet = []
tsheet = []

'''
Open the spreadsheet to the training sheet
'''
with open('./data_NESARC_tr.csv/data_NESARC_tr.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        sheet += [[row[8], row[10], row[21], row[22], row[26], row[28], row[2989], row[49], row[59], row[62], row[70],\
        row[88], row[102], row[114], row[118], row[122], row[124], row[156], row[168], row[169], row[170], row[171]]]

#Open the spreadsheet to the test sheet
with open('./data_NESARC_ts.csv/data_NESARC_ts.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        tsheet += [[row[8], row[10], row[21], row[22], row[26], row[28], row[2989], row[49], row[59], row[62], row[70],\
        row[88], row[102], row[114], row[118], row[122], row[124], row[156], row[168], row[169], row[170], row[171]]]


#format and create the data
TX = []
TY = []
X = []
Y = []

'''
Normalize the data
'''
def normalize(X, Y, s):
    for i in s:
        var42 = [0]*9
        var42[int(i[0]) - 1] = 1
        var48 = (float(i[1]) - 1.0)/17.0
        var63 = abs(int(i[2]) - 2)
        var64 = abs(int(i[3]) - 2)
        var68 = (float(i[4]) - 18.0)/72.0
        var79 = abs(int(i[5]) - 2)
        var3673 = [0]*5
        var3673[int(i[6]) - 1] = 1
        var114 = [0]*6
        var114[int(i[7]) - 1] = 1
        var131 = (float(i[8]) - 1.0)/13.0
        var136 = abs(int(i[9]) - 2)
        var144 = abs(int(i[10]) - 2)
        var163 = abs(int(i[11]) - 2)
        var196 = (float(i[12]) - 1.0)/20.0
        var217 = abs(int(i[13]) - 2)
        var226 = abs(int(i[14]) - 2)
        var230 = abs(int(i[15]) - 2)
        var232 = 0.0
        if (int(i[16]) != 9):
            var232 += (float(i[16]) - 1.0)/4.0
        elif (int(i[16]) == 9):
            var232 = 2.0/4.0

        var294 = 0
        if (int(i[17]) != 9):
            var294 = abs(int(i[17]) - 2)

        varHeight = 0.0
        if (int(i[18]) != 99):
            varHeight += float(i[18]) * 12
            if (int(i[19]) != 99):
                varHeight += float(i[19])
            varHeight /= 95.0

        var310 = 0.0
        if (int(i[20]) != 999):
            var310 = (float(i[20]) - 62.0)/438.0

        X += [[1] + var42 + [var48] + [var63] + [var64] + [var68] + [var79] + var3673 + var114 + [var131] + [var136] + \
        [var144] + [var163] + [var196] + [var217] + [var226] + [var230] + [var232] + [var294] + [varHeight] + [var310]]
        #prediction variable
        var313 = abs(int(i[-1]) - 2)
        Y += [var313]

'''
Run logistic regression across all data
'''
def regression(X, W, Y, lambd, scalar, step, it):
    count = 0
    while (count < it):
        g = np.empty([1, X.shape[1]])
        total = 0.0
        for i in range(0, 30000):
            new = (1.0/(1.0 + math.exp(-1 * np.dot(X[i], W.transpose())[0, 0]))) + scalar
            g = np.add(g, np.dot((new - Y[i]), X[i]))
            #check if the value was correctly guessed
            b = 1 if (new > 0.5) else 0
            if b == Y[i]: total += 1
        #apply lambda only if it's greater than 0 to avoid performance hit
        if lambd != 0:
            g = np.add(g, np.dot(lambd, W))
        W = np.subtract(W, np.dot(step/30000.0, g))
        count += 1
        #prints only every 10 counts to avoid performance hits
        if count % 10 == 0:
            e = total/300
            print "correct: " + str(e) + "%, " + "error: " + str(100 - e) + "%, iteration: " + str(count)
    return W

'''
Test the weights using the sigmoid function
'''
def test(X, W, Y):
    count = 0
    for i in range(0, X.shape[0]):
        new = (1.0/(1.0 + math.exp(-1 * np.dot(X[i], W.transpose())[0, 0])))
        a = 0
        if new > 0.5:
            a = 1
            print "A"
        if a == Y[i]:
            count += 1
    e = float(count)/float(X.shape[0]) * 100
    print "Test - correct: " + str(e) + "%, " + "error: " + str(100 - e)

def main(X, Y, TX, TY, argv, argc):


    #normalize the
    normalize(X, Y, sheet)
    W = [0.0]*len(X[0])
    #convert to numpy matrix
    W = np.matrix(W)
    X = np.matrix(X)
    #run the regression
    weights = []
    cont = False
    if argc == 1:
        weights = regression(X, W, Y, 0, 0, 1, 700)
        cont = True

    elif argc > 1 and argc == 5:
        weights = regression(X, W, Y, float(argv[1]), float(argv[2]), float(argv[3]), int(argv[4]))
        cont = True

    else:
        print "Usage: python logistic_regression.py <lambda> <nu> <eta> <iterations>"
        cont = False

    if cont:
        normalize(TX, TY, tsheet)
        TX = np.matrix(TX)
        test(TX, weights, TY)
        print "Final weights: \n" + str(weights)

if __name__ == "__main__":
    main(X, Y, TX, TY, sys.argv, len(sys.argv))
