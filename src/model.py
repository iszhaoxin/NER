from functions import *
import os, sys
sys.path.append("../data")
import numpy as np
from mylib.texthelper import *
import dataset

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import itertools,re

# sent : [<dataset.Token>1,<dataset.Token>2, ...... ,<dataset.Token>n]
# i : 第i个位置
def word2features(sent, i):
    word = sent[i].text
    token = sent[i]
    features = {}
    features['bias'] = 1.0
    # if hasdigtal(token):
    #     if token.label == "None":
    #         print(token.text)
    for j in range(-2,3):
        try:
            assert(i+j>=0)
            token = sent[i+j]
        except:
            if j<0:
                features[str(j)+"BOS"] = True
            elif j>0:
                features[str(j)+"EOS"] = True
            continue
        prefix = str(j)+':'
        # features[prefix+'shape'] = shape(token)
        features[prefix+'postag'] = token.tag
        # features[prefix+'isword'] = isword(token)
        features[prefix+'periodic'] = periodic(token)
        features[prefix+'hasalpha'] = hasalpha(token)
        features[prefix+'hasAlpha'] = hasAlpha(token)
        features[prefix+'hasdigtal'] = hasdigtal(token)
        features[prefix+'hasOther'] = hasOther(token)
        features[prefix+'variable'] = variable(token)
        features[prefix+'quantity'] = quantity(token)
        features[prefix+'isOperator'] = isOperator(token)
        features[prefix+'isdigit'] = isdigit(token)
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [token.label for token in sent]

def sent2tokens(sent):
    return [token.text for token in sent]



if __name__ == "__main__":
    train_dir_path = "../data/dataset/train/"
    train_data = dataset.Dataset(train_dir_path)
    test_dir_path = "../data/dataset/test/"
    test_data = dataset.Dataset(test_dir_path)

    X_test,X_train,y_test,y_train = [],[],[],[]

    for ins,sentences in train_data.nerAnnoIter():
        X_train += [sent2features(s) for s in sentences]
        y_train += [sent2labels(s) for s in sentences]
    for ins,sentences in test_data.nerAnnoIter():
        X_test += [sent2features(s) for s in sentences]
        y_test += [sent2labels(s) for s in sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0,
        c2=0,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    print(labels)

    y_pred = crf.predict(X_test)
    # y_pred = crf.predict(X_train)
    metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    # metrics.flat_f1_score(y_train, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    # print(metrics.flat_classification_report(y_train, y_pred, labels=sorted_labels, digits=3))
