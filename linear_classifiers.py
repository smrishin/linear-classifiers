import sys
import random
import math
#### FUNCTIONS #########

###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0, len(u), 1):
        dp += u[i]*v[i]
    return dp

###################
## Standardize the code here: divide each feature of each 
## datapoint by the length of each column in the training data
## return [traindata, testdata]
###################
def standardize_data(traindata, testdata):
 
    for i in range(0, len(traindata[0]), 1):
        fact = 0
        for j in range(0, len(traindata), 1):
            fact += math.pow(traindata[j][i], 2)
        fact = math.sqrt(fact)
        if fact != 0:
            for a in range(0, len(traindata), 1):
                traindata[a][i] /= fact
            for b in range(0, len(testdata), 1):
                testdata[b][i] /= fact
    return [traindata, testdata]
        
eta = 0.001
stop = 0.001
lamda = 0.01

###################
## Solver for least squares (linear regression)
## return [w, w0]
###################
def least_squares(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01,0.01))
    old_error = 0
    product=0
    while True:
        delta = []
        er = 0
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                for j in range(0, len(traindata[0]), 1):
                    difference = trainlabels[i] - product
                    delta[j] += difference*(traindata[i][j])
        for j in range(0, len(traindata[0]), 1):
            w[j] += eta * delta[j]                    # w 
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                difference = trainlabels[i] - dotproduct(w,traindata[i])
                er += math.pow(difference, 2)                   # error
        print(str(er))
        w0 = [len(w) - 1]                    # w0 
        if(abs(old_error - er) <= stop):
            break
        old_error = er
        
    return[w,w0]

###################
## Solver for regularized least squares (linear regression)
## return [w, w0]
###################
def least_squares_regularized(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01, 0.01))
    old_error = 0
    product = 0
    while True:
        delta= []
        er = 0;
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                for j in range(0, len(traindata[0]), 1):
                    difference = trainlabels[i] - product
                    reg = lamda * w[j]
                    delta[j] += difference * (traindata[i][j]) + reg
        for j in range(0, len(traindata[0]), 1):
            w[j] += eta * delta[j]                       # w 
        for i in range(0, len(traindata), 1):
            w2 = 0
            for j in range(0, len(w)):
                w2  += math.pow(w[j], 2)
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                difference = trainlabels[i] - product
                reg = lamda * w2                                #regularized
                er += math.pow(difference, 2) + reg             #error
        print(str(er))
        w0 = [len(w) - 1]                                # w0       
        if (abs(old_error - er) <= stop):
            break;
        old_error = er;

    return [w, w0]

###################
## Solver for hinge loss
## return [w, w0]
###################
def hinge_loss(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01,0.01)) 
    product = 0
    old_error = len(traindata) * 10
    while True:      
        delta = []
        er = 0
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                check = (trainlabels[i]) * product
                for j in range(0, len(traindata[0]), 1):
                    if(check < 1):
                        q = (trainlabels[i])*(traindata[i][j])
                        delta[j] += -1 * q
        for j in range(0, len(traindata[0]), 1):
            w[j] -= eta * delta[j]                  # w
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w,traindata[i])
                p = trainlabels[i] * product
                er += max(0, 1 - p)                 #error/hinge loss
        print(str(er))
        w0 = [len(w) - 1]                           # w0
        if(abs(old_error - er) <= stop):
            break
        old_error = er
        
    return[w,w0]

###################
## Solver for regularized hinge loss
## return [w, w0]
###################
def hinge_loss_regularized(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01, 0.01))
    product = 0
    old_error = len(traindata) * 10
    while True:
        delta = []
        er = 0
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                check = (trainlabels[i]) * product
                for j in range(0, len(traindata[0]), 1):
                    if (check < 1):
                        q = (trainlabels[i]) * (traindata[i][j])
                        reg = lamda * w[j]
                        delta[j] += - 1 * q + reg
        for j in range(0, len(traindata[0]), 1):
            w[j] -= eta * delta[j]                  # w
        for i in range(0, len(traindata), 1):
            w2 = 0
            for j in range(0, len(w)):
                w2 += w[j] ** 2
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                p = trainlabels[i] * product
                reg =  0.5 * lamda * w2                     #regularized
                er += max(0, 1 - p) + reg                   # error/hinge loss
        print(str(er))
        w0 = [len(w) - 1]                                   # w0
        if (abs(old_error - er) <= stop):
            break
        old_error = er

    return [w, w0]


###################
## Solver for logistic regression
## return [w, w0]
###################
def logistic_loss(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01,0.01))
    product = 0
    old_error = len(traindata) * 10
    while True:
        delta = []
        er = 0
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)    
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                q = 1 / (1+ (math.exp(-1 * product)))
                ex = trainlabels[i] - q
                for j in range(0, len(traindata[0]), 1):
                        delta[j] += ex * (traindata[i][j])
        for j in range(0, len(traindata[0]), 1):
            w[j] += eta * delta[j]                           # w
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w,traindata[i])
                p = trainlabels[i] * product
                er += math.log(1 + math.exp((-1 * p)))      #error/logistic loss
        print(str(er))
        w0 = [len(w) - 1]                                   # w0
        if(abs(old_error - er) <= stop):
            break
        old_error = er
        
    return[w,w0]

    
###################
## Solver for adaptive learning rate hinge loss
## return [w, w0]
###################
def hinge_loss_adaptive_learningrate(traindata, trainlabels):
    w = []
    w0 = []
    for j in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01,0.01))
    eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
    best_eta = 1
    prevob = 1000000000
    bestob = 1000000000000
    ob = prevob - 10
    product = 0
    delta = []
    delta_len = 0
    for i in range(0, len(traindata[0]), 1):
        delta.append(0)
    while (prevob - ob > stop):
        prevob = ob
        for j in range(0, len(traindata[0]), 1):
            delta[j] = 0
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w,traindata[i])
                check = (trainlabels[i]) * product
                for j in range(0, len(traindata[0]), 1):
                    if (check < 1):
                        q =trainlabels[i]*traindata[i][j]
                        delta[j] += -1 * q
        for j in range(0, len(traindata[0]), 1):
            delta_len += delta[j]**2
        delta_len = math.sqrt(delta_len)
        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]
            for j in range(0, len(traindata[0]), 1):
                w[j] += eta * delta[j]              # updating w
            er = 0
            for i in range(0, len(traindata), 1):
                if (trainlabels[i] != None):
                    product = dotproduct(w,traindata[i])
                    if(trainlabels[i] * product < 1):
                        er += 1 - (trainlabels[i] * product)
            ob = er
            if(ob < bestob):
                best_eta = eta
                bestob = ob
            for j in range(0, len(traindata[0]), 1):
                w[j] -= eta * delta[j]
        eta = best_eta
        for j in range(0, len(traindata[0]), 1):
                w[j] += eta * delta[j]
        er = 0;
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w,traindata[i])
                if((trainlabels[i] * product) < 1):
                    er += 1 - (trainlabels[i] * product)
        ob = er     
        print(str(er))
        w0 = [len(w) - 1]               # w0
        
    return[w,w0]

    
    
#### MAIN #########

###################
#### Code to read train data and train labels
###################
trainfile = sys.argv[1]
testfile = sys.argv[2]
traindata = []
trainlabels = []
testdata = []
testlabels = []

f = open(trainfile) 
line = f.readline()
while (line != ''):
    values = line.split()
    y = []
    for val in range(1, len(values)):
        y.append(float(values[val]))
    traindata.append(y)
    trainlabels.append(float(values[0]))
    line = f.readline()

###################
#### Code to test data and test labels
#### The test labels are to be used
#### only for evaluation and nowhere else.
#### When your project is being graded we
#### will use 0 label for all test points
###################
f = open(testfile) 
line = f.readline()
while (line != ''):
    values = line.split()
    y = []
    for val in range(1, len(values)):
        y.append(float(values[val]))
    testdata.append(y)
    testlabels.append(float(values[0]))
    line = f.readline()




#standardization
[traindata, testdata] = standardize_data(traindata, testdata)

# ###################
# #### Classify unlabeled points
# ##################

# Prediction
'''def printOutput(w, w0, testdata, name):
OUT = open(name, 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
'''
################################
#Least Squares
################################
print("1")
print("Computing w,w0 : Least Square.......")
[w,w0] = least_squares(traindata, trainlabels)
print("Printing Prediction to least_squares_prediction.txt")
#printOutput(w, w0, testdata, "least_squares_prediction.txt")
OUT = open(r"result\least_squares_prediction.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
print("DONE : Least Square")
OUT.close()

################################
#Regularized Least Square
################################
print("2")
print("Computing w,w0 : Regularized Least Square.......")
[w,w0] = least_squares_regularized(traindata, trainlabels)
print("Printing Prediction to reg_least_squares_prediction.txt")
#printOutput(w, w0, testdata, "reg_least_squares_prediction.txt")
OUT = open(r"result\reg_least_squares_prediction.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
print("DONE : Regularized Least Square")
OUT.close()

################################
#Hinge Loss
################################
print("3")
print("Computing w,w0 : Hinge Loss.......")
[w,w0] = hinge_loss(traindata, trainlabels)
print("Printing Prediction to hinge_predictions.txt")
#printOutput(w, w0, testdata, "hinge_predictions.txt")
OUT = open(r"result\hinge_predictions.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
print("DONE : Hinge Loss")
OUT.close()

################################
#Regularized Hinge Loss
################################
print("4")
print("Computing w,w0 : Regularized Hinge Loss.......")
[w,w0] = hinge_loss_regularized(traindata, trainlabels)
print("Printing Prediction to regularized_hinge_prediction.txt")
#printOutput(w, w0, testdata, "regularized_hinge_prediction.txt")
OUT = open(r"result\regularized_hinge_prediction.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
print("DONE : Regularized Hinge Loss")
OUT.close()

################################
#Logistric Regression
################################
print("5")
print("Computing w,w0 : Logistic Regression.......")
[w,w0] = logistic_loss(traindata, trainlabels)
print("Printing Prediction to logistic_prediction.txt")
OUT = open(r"result\logistic_prediction.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
   product = dotproduct(w, testdata[i]);
   if (product < 0.5):
       print("0", i, file = OUT);
   else:
       print("1",i, file = OUT);
print("DONE : Logistic Regression")
OUT.close()

################################
#Adaptive Hinge Loss
################################
print("6")
print("Computing w,w0 : Adaptive Hinge Loss.......")
[w,w0] = hinge_loss_adaptive_learningrate(traindata, trainlabels)
print("Printing Prediction to Adaptive_eta_hinge_prediction.txt")
#printOutput(w, w0, testdata, "adaptive_eta_hinge_prediction.txt")
OUT = open(r"result\adaptive_eta_hinge_prediction.txt", 'w')
rows = len(testdata)
for i in range(0, rows, 1):
    product = dotproduct(w, testdata[i]);
    if (product < 0):
        print("0", i, file = OUT);
    else:
        print("1",i, file = OUT);
print("DONE : Adaptive Hinge Loss")
OUT.close()

###################
## Optional for testing on toy data
## Comment out when submitting project
###################
# print(w)
# wlen = math.sqrt(w[0]**2 + w[1]**2)
# dist_to_origin = abs(w[2])/wlen
# print("Dist to origin=",dist_to_origin)
#
# wlen=0
# for i in range(0, len(w), 1):
# 	wlen += w[i]**2
# wlen=math.sqrt(wlen)
# print("wlen=",wlen)
#
