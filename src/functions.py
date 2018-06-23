import os, sys, re

from functions import *
import os, sys
sys.path.append("../data")
import numpy as np
from mylib.texthelper import *
import dataset

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No',
 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
 'Lv', 'Ts', 'Og', 'Uue')
variable_table = ('x', 'y', 'd')
quantity_table = ('zT','ZT','K', 'μ','F','C','Ω','mΩ','mA','mV','mW','μΩ','μV','μA','μW','°','∼')
operator_table = ['x','±','-','\*','>','<','/','=','≤','≥']
def isword(token):
    if re.fullmatch('X?x*',token.shape) is not None:
        return True
    return False
def hasAlpha(token):
    if "X" in token.shape:
        return True
    return False
def hasalpha(token):
    if "x" in token.shape:
        return True
    return False
def hasdigtal(token):
    if "d" in token.shape:
        return True
    return False

def hasOther(token):
    if re.sub('x|X|d','',token.shape) != '':
        return True
    return False

def shape(token):
    sign = str(int("X" in token.shape))
    sign += str(int("x" in token.shape))
    sign += str(int("d" in token.shape))
    sign += str(int(re.sub('x|X|d','',token.shape) != ''))
    return sign

def periodic(token):
    if isword(token) == False:
        for i in periodic_table:
            if i in token.text and len(i) > 1:
                return True
    return False

def isdigit(token):
    rgx_operators = "("+'|'.join(operator_table)+")?"+'\d+\.+\d+'
    rgx = re.compile(rgx_operators)
    bool = rgx.fullmatch(token.text) is not None
    # if bool == True:
    #     print(token.text,token.label)
    return rgx.fullmatch(token.text) is not None

def variable(token):
    if len(token.text) == 1 and token.shape == 'x':
        return True
    return False

def quantity(token):
    for i in quantity_table:
        if i in token.text:
            if i in token.text:
                return True
    return False
def isOperator(token):
    if len(token.text) == 1 and token.shape != 'x'and token.shape != 'X'and token.shape != 'd':
        return True
    return False


if __name__ == "__main__":
    train_dir_path = "../data/dataset/train/"
    train_data = dataset.Dataset(train_dir_path)
    test_dir_path = "../data/dataset/test/"
    test_data = dataset.Dataset(test_dir_path)

    X_test,X_train,y_test,y_train = [],[],[],[]

    for ins,sentences in train_data.nerAnnoIter():
        for sentence in sentences:
            for w in sentence:
                if w.label == 'Value':
                    print(w.text)
                # if hasdigtal(w):
                #     print(w.text)
        # X_test += [sent2features(s) for s in sentences]
        # y_test += [sent2labels(s) for s in sentences]
