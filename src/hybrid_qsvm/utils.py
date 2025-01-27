# Created by Dennis Willsch (d.willsch@fz-juelich.de) 
# Modified by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 

import sys
import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,accuracy_score,auc
from hybrid_qsvm.load_data import *
from hybrid_qsvm.QgSVM_utils import QgSVM


np.set_printoptions(precision=4, suppress=True)

def kernel(xn, xm, gamma=-1): # here (xn.shape: NxD, xm.shape: ...xD) -> Nx...
    krenel = QgSVM(int(gamma))
    kernel = krenel.get_kernel()
    return kernel(xn, xm)

# B = base
# K = number of qubits per alpha

# decode binary -> alpha
def decode(binary, B=10, K=3):
    N = len(binary) // K
    Bvec = B ** np.arange(K)
    return np.fromiter(binary,float).reshape(N,K) @ Bvec

# encode alpha -> binary with B and K (for each n, the binary coefficients an,k such that sum_k an,k B**k is closest to alphan)
def encode(alphas, B=10, K=3):
    N = len(alphas)
    Bvec = B ** np.arange(K) # 10^0 10^1 10^2 ...
    allvals = np.array(list(map(lambda n : np.fromiter(bin(n)[2:].zfill(K),float,K), range(2**K)))) @ Bvec # [[0,0,0],[0,0,1],...] @ [1, 10, 100]
    return ''.join(list(map(lambda n : bin(n)[2:].zfill(K),np.argmin(np.abs(allvals[:,None] - alphas), axis=0))))

def encode_as_vec(alphas, B=10, K=3):
    return np.fromiter(encode(alphas,B,K), float)

def seqs_to_onehots(seqs): # from ../utils.py
    return np.asarray([np.asarray([[1 if bp == letter else 0 for letter in 'ACGT'] for bp in seq]).flatten() for seq in seqs])

def loadraw(key='mad50'): # key = 'mad50', 'myc99', ... basically from do-svm.py
    data = np.genfromtxt(f'data/intensities-{key[:3]}filtered', dtype=None, names=True, encoding=None, usecols=(0,1))
    phis = seqs_to_onehots(data['sequence'])
    X = 2*phis - 1  
    ys = data['log_mean']

    percentile = float(key[3:])
    theta_percentile = int(len(data) * percentile / 100.)
    theta_idx = np.argpartition(ys, theta_percentile)[theta_percentile]
    theta = ys[theta_idx]
    labels = np.sign(ys - theta)
    labels[theta_idx] = 1 # corner case is counted as 1 (b/c we have >= theta in mlr)

    return X, labels

#def loaddataset(datakey='mad50p2calibtrain0'):
#    dataset = np.loadtxt('data/datasets/'+datakey, dtype=float, skiprows=1)
#    return dataset[:,2:], dataset[:,1]  # data, labels

def loaddataset(datakey):
    dataset = np.loadtxt(datakey, dtype=float, skiprows=1)
    return dataset[:,2:], dataset[:,1]  # data, labels

def save_json(filename, var):
    with open(filename,'w') as f:
        f.write(str(json.dumps(var, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)))

def eval_classifier(x, alphas, data, label, gamma, b=0): # evaluates the distance to the hyper plane according to 16.5.32 on p. 891 (Numerical Recipes); sign is the assigned class; x.shape = ...xD
    return np.sum((alphas * label)[:,None] * kernel(data, x, gamma), axis=0) + b

def eval_offset_avg(alphas, data, label, gamma, C, useavgforb=True): # evaluates offset b according to 16.5.33
    cross = eval_classifier(data, alphas, data, label, gamma) # cross[i] = sum_j aj yj K(xj, xi) (error in Numerical Recipes)
    if useavgforb:
        return np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))
    else:  # this is actually not used, but we did a similar-in-spirit implementation in eval_finaltraining_avgscore.py
        if np.isclose(np.sum(alphas * (C-alphas)),0):
            print('no support vectors found, discarding this classifer')
            return np.nan
        bcandidates = [np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))]  # average according to NR should be the first candidate
        crosssorted = np.sort(cross)
        crosscandidates = -(crosssorted[1:] + crosssorted[:-1])/2  # each value between f(xi) and the next higher f(xj) is a candidate
        bcandidates += sorted(crosscandidates, key=lambda x:abs(x - bcandidates[0]))  # try candidates closest to the average first
        bnumcorrect = [(label == np.sign(cross + b)).sum() for b in bcandidates]
        return bcandidates[np.argmax(bnumcorrect)]

def eval_acc_auroc_auprc(label, score):  # score is the distance to the hyper plane (output from eval_classifier)
    precision,recall,_ = precision_recall_curve(label, score)
    return accuracy_score(label,np.sign(score)), roc_auc_score(label,score), auc(recall,precision)



################ This I/O functions are provided by http://hyperlabelme.uv.es/index.html ################ 

def dataread(filename):
    lasttag = 'description:'
    # Open file and locate lasttag
    f = open(filename, 'r')
    nl = 1
    for line in f:
        if line.startswith(lasttag): break
        nl += 1
    f.close()

    # Read data
    data = np.loadtxt(filename, delimiter=',', skiprows=nl)
    Y = data[:, 0]
    X = data[:, 1:]
    # Separate train/test
    Xtest = X[Y < 0, :]
    X = X[Y >= 0, :]
    Y = Y[Y >= 0, None]

    return X, Y, Xtest


def datawrite(path,method, dataset, Yp):
    filename = '{0}{1}_predictions.txt'.format(path, dataset)
    res = True
    try:
        with open(filename, mode='w') as f:
            f.write('{0} {1}'.format(method, dataset))
            for v in Yp:
                f.write(' {0}'.format(str(v)))
            f.write('\n')
    except Exception as e:
        print('Error', e)
        res = False
    return res

################ 


def write_samples(X, Y,path): 
    f = open(path,"w+") 
    f.write("id label data \n") 
    for i in range(0,X.shape[0]):
        f.write(str(i)+" ")
        if(Y[i]==1):
            f.write("1 ")
        else:
            f.write("-1 ")
        for j in range(0,X.shape[1]):
            f.write(str(X[i,j])+" ")
        f.write("\n") 
    f.close() 