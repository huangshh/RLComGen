import logging, pickle, os, random, math, sys
import getopt
import numpy as np
from time import time
from utils import *
from bleu import *
import pickle
import torch
from prefetch_generator import BackgroundGenerator

method = ""
pretrain = ""
start = -1
options, args = getopt.getopt(sys.argv[1:],"",['method=','start=','pretrain='])
for name, value in options:
    if name == "--method":
        method = value
    elif name == "--start":
        start = int(value)
    elif name == "--pretrain":
        pretrain = value

dict_path = "./dataset/"
ast_w2i = pickle.load(open(os.path.join(dict_path,"ast_w2i.pkl"),"rb"))
ast_i2w = pickle.load(open(os.path.join(dict_path,"ast_i2w.pkl"),"rb"))
nl_w2i = pickle.load(open(os.path.join(dict_path,"nl_w2i.pkl"),"rb"))
nl_i2w = pickle.load(open(os.path.join(dict_path,"nl_i2w.pkl"),"rb"))
api_w2i = pickle.load(open(os.path.join(dict_path,"api_w2i.pkl"),"rb"))
api_i2w = pickle.load(open(os.path.join(dict_path,"api_i2w.pkl"),"rb"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_filename = "./{}/log".format(method)
fh = logging.FileHandler(log_filename)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

dataset_path = "./dataset/"
valid_ast = os.path.join(dataset_path,"valid.token-nl2.token")
valid_nl = os.path.join(dataset_path,"valid.token-nl2.nl2")
valid_api = os.path.join(dataset_path,"valid.token-nl2.api")

if method == "deepcom":
    from deepcom.deepcom import *
    from deepcom.config import Config
    config = Config()
    model = Deepcom(config)
elif method == "TLCodeSum":
    from TLCodeSum.TLCodeSum import *
    from TLCodeSum.config import Config
    config = Config()
    model = TLCodeSum(config)
elif method == "RLCom":
    from RLCom.RLCom import *
    from RLCom.config import Config
    config = Config()
    model = RLCom(config)
elif method == "RLComAPI":
    from RLComAPI.RLComAPI import *
    from RLComAPI.config import Config
    config = Config()
    model = RLComAPI(config)

if start != -1:
    model.load("./{}/save/{}.pkl".format(method,start))
    model.set_trainer()
        
has_valid = 0
valid_dataset = BackgroundGenerator(get_batch(valid_ast,valid_nl,config.EVAL_BATCH_SIZE,valid_api), 1)
corpus = []
for x_raw,y_raw,z_raw in valid_dataset:
    has_valid += 1
    x = [[ast_w2i[token] if token in ast_w2i else ast_w2i['<UNK>'] for token in line[:config.MAX_SEQ_LEN]] for line in x_raw]
    if method == "TLCodeSum" or method == "RLComAPI":
        z = [[api_w2i[token] if token in api_w2i else api_w2i['<UNK>'] for token in line[:config.MAX_API_LEN]] for line in z_raw]   
        x = [x,z]             
    preds = model(x,False)
    for i in range(len(preds)):
        pred_ = [nl_i2w[token] if token in nl_i2w else "<UNK>" for token in preds[i]]
        pred_ = [token.lower() for token in pred_]
        corpus += [pred_]

with open('pred.txt','w') as f:
    for pred in corpus:
        f.write(" ".join(pred))
        f.write("\n")