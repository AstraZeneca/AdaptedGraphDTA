from math import sqrt
import numpy as np
from scipy import stats

def rmse(y,f):
    return sqrt(((y - f)**2).mean(axis=0))

def mse(y,f):
    return ((y - f)**2).mean(axis=0)

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
