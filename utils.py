import nltk
from nltk.translate.bleu_score import *
from time import time
import collections
from collections import defaultdict
import numpy as np


def get_time(t):
    t = int(t)
    h = t//3600
    m = t%3600//60
    s = t%60
    h = str(h)
    if m<10:
        m = '0'+str(m)
    else:
        m = str(m)
    if s<10:
        s = '0'+str(s)
    else:
        s = str(s)
    return h+":"+m+":"+s

def get_batch(ast_file,nl_file,batch_size,api_file=None):
    if api_file==None:
        x_raw = []
        y_raw = []
        with open(ast_file) as f1, open(nl_file) as f2:
            for x,y in zip(f1,f2):
                x_ = x.strip().split(" ")
                y_ = y.strip().split(" ")
                x_raw.append(x_)
                y_raw.append(y_)
                if len(x_raw)>=batch_size:
                    yield x_raw,y_raw
                    x_raw = []
                    y_raw = []
        
        if len(x_raw)>0:
            yield x_raw,y_raw
    else:
        x_raw = []
        y_raw = []
        z_raw = []
        with open(ast_file) as f1, open(nl_file) as f2, open(api_file) as f3:
            for x,y,z in zip(f1,f2,f3):
                x_ = x.strip().split(" ")
                y_ = y.strip().split(" ")
                z_ = z.strip().split(" ")
                x_raw.append(x_)
                y_raw.append(y_)
                z_raw.append(z_)
                if len(x_raw)>=batch_size:
                    yield x_raw,y_raw,z_raw
                    x_raw = []
                    y_raw = []
                    z_raw = []
        
        if len(x_raw)>0:
            yield x_raw,y_raw,z_raw