import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import model_selection
import time

#
#fig, ax = plt.subplots()
#line, = ax.plot([], [], lw=2)
# ax.grid()

primes = [1, 2]


def primeGen(N):
    nums = [i for i in range(1, N)]
    for i in nums:
        facs = []
        for j in range(1, i):
            if(i % j == 0):
                facs.append(j)
        if(len(facs) == 1):
            primes.append(i)

    primes.sort()


def genYval(x_arr):
    y_arr = []
    for i in x_arr:
        if(i in primes):
            y_arr.append(1)
        else:
            y_arr.append(0)
    return y_arr


def Exp1(Xt, Yt, Xv, Yv):

    lrc = LogisticRegression()
    lrc.fit(Xt, Yt)
    # Predict
    pred = lrc.predict(Xv)
    return pred, accuracy_score(Yv, pred)


scores = []
primeGen(100)
primes = list(set(primes))
X = [x for x in range(1, 100)]
y = genYval(X)

print np.size(X)
print np.size(y)

Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=0)

Xtrain = np.flatten(Xtrain)
# svm1 = SVC()
# svm1.fit(Xtrain, Ytrain)
# y_pred = svm1.predict(Xtest)
print(Exp1(Xtrain, Ytrain, Xtest, Ytest))
