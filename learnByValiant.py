import itertools
import pickle
import numpy as np
import pandas as pd
import sys

from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score

class learnByValiant:
    def __init__(self,k = 3,clauses = []):
        self.model = clauses
        self.k = k # k is the arity of the learnt CNF formula
    def fit(self,X,y):
        n = len(X.columns)
        print('n = '+repr(n))
        positiveIndex = []
        for index, row in y.iterrows():
            if y.loc[index]['f'] == 1:
                positiveIndex.append(index)
        print('ratio of positive assignments: '+repr(len(positiveIndex))+' / '+repr(len(X)))
        clauseCount = 0
        possibleClauses = self.possibleClauses(n)
        numPossibleClauses = len(possibleClauses)
        for clause in self.possibleClauses(n):
            clauseCount += 1
            if self.checkClause(clause,X,positiveIndex) == True:
                self.model.append(clause)
            #print("learning process (# clauses checked): "+repr(clauseCount)+" / "+repr(numPossibleClauses)+" ("+repr(clauseCount * 100.0 / numPossibleClauses)+"%)")
        print('learned a ' +repr(self.k) + '-CNF formula by Valiants algorithm with '+repr(len(self.model))+' clauses.')
        print('ratio of positive assignments: ' + repr(len(positiveIndex)) + ' / ' + repr(len(X)))
        return self.model
    def get_params(self,deep = False):
        return {'k':self.k,'clauses':self.model}
    def printModel(self):
        print('printing model with '+repr(len(self.model)) + ' clauses')
        for clause in self.model:
            print(clause)
    def predict(self, X):
        predictAns = []
        for index, row in X.iterrows():
            flag = 1
            for clause in self.model:
                if self.evaluateClause(clause, row) == 0:
                    predictAns.append(0)
                    flag = 0
                    break
            if flag == 1:
                predictAns.append(1)
        print(predictAns)
        return predictAns
    def evaluateClause(self,clause,assignment): # assignment is a -1/+1 string
        for i in clause:
            if (2 * assignment[abs(i)-1] - 1 ) * i > 0: # the literal is satisfied
                return 1
        return 0
    def checkClause(self,clause,X,positiveIndex):
        for index in positiveIndex:
            row = X.loc[index]
            #print('encounter a positive assignment at index '+repr(index))
            if self.evaluateClause(clause, row) == 0:
                return False
        return True
    def possibleClauses(self,n):
        combs = itertools.combinations(range(1,n+1),self.k)
        allClauses = []
        signs = list(itertools.product([-1, 1], repeat=self.k))
        for positiveClause in combs:
            for signTuple in signs:
                allClauses.append([positiveClause[i] * signTuple[i] for i in range(self.k)])
        return allClauses

filename = sys.argv[1]
kfolds = int(sys.argv[2])
CNFArity = int(sys.argv[3])
scoring = ['accuracy', 'f1_macro']
with open(filename,'rb') as f:
    data = pickle.load(f)
    data_x, data_y = data.iloc[:, :-1], data.iloc[:, [-1]]
    kf = model_selection.KFold(n_splits=kfolds, random_state=None)
    learner = learnByValiant(CNFArity)
    scores = cross_validate(learner, data_x, data_y, cv=kf, scoring=scoring)
    acc, f1 = np.mean(scores['test_accuracy']), np.mean(scores['test_f1_macro'])
    print('acc = '+repr(acc)+', f1 = '+repr(f1))
