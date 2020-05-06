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
datatype = "ast"
options, args = getopt.getopt(sys.argv[1:],"",['method=','start=','pretrain=','datatype='])
for name, value in options:
    if name == "--method":
        method = value
    elif name == "--start":
        start = int(value)
    elif name == "--pretrain":
        pretrain = value
    elif name == "--datatype":
        datatype = value

dict_path = "./dataset/"
if datatype == "ast":
    ast_w2i = pickle.load(open(os.path.join(dict_path,"ast_w2i.pkl"),"rb"))
    ast_i2w = pickle.load(open(os.path.join(dict_path,"ast_i2w.pkl"),"rb"))
else:
    ast_w2i = pickle.load(open(os.path.join(dict_path,"code_w2i.pkl"),"rb"))
    ast_i2w = pickle.load(open(os.path.join(dict_path,"code_i2w.pkl"),"rb"))
nl_w2i = pickle.load(open(os.path.join(dict_path,"nl_w2i.pkl"),"rb"))
nl_i2w = pickle.load(open(os.path.join(dict_path,"nl_i2w.pkl"),"rb"))
api_w2i = pickle.load(open(os.path.join(dict_path,"api_w2i.pkl"),"rb"))
api_i2w = pickle.load(open(os.path.join(dict_path,"api_i2w.pkl"),"rb"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_filename = "./{}/{}_log".format(method,datatype)
fh = logging.FileHandler(log_filename)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

dataset_path = "./dataset/"

if datatype == "ast":
    train_ast = os.path.join(dataset_path,"train.token-nl2.token")
    valid_ast = os.path.join(dataset_path,"valid.token-nl2.token")
    test_ast = os.path.join(dataset_path,"test.token-nl2.token")
else:
    train_ast = os.path.join(dataset_path,"train.token-nl2.code2")
    valid_ast = os.path.join(dataset_path,"valid.token-nl2.code2")
    test_ast = os.path.join(dataset_path,"test.token-nl2.code2")
train_nl = os.path.join(dataset_path,"train.token-nl2.nl2")
valid_nl = os.path.join(dataset_path,"valid.token-nl2.nl2")
test_nl = os.path.join(dataset_path,"test.token-nl2.nl2")
train_api = os.path.join(dataset_path,"train.token-nl2.api")
valid_api = os.path.join(dataset_path,"valid.token-nl2.api")
test_api = os.path.join(dataset_path,"test.token-nl2.api")

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
    model.load("./{}/save/{}_{}.pkl".format(method,datatype,start))
    model.set_trainer()

save_path = './{}/save/'.format(method)
if not os.path.exists(save_path):
    os.mkdir(save_path)

random.seed(config.SEED)
best_bleu = 0.
t0 = time()
for epoch in range(start+1,config.EPOCH):
        has_trained = 0
        train_dataset = BackgroundGenerator(get_batch(train_ast,train_nl,config.TRAIN_BATCH_SIZE,train_api), 1)
        for x_raw,y_raw,z_raw in train_dataset:
            has_trained += 1
            x = [[ast_w2i[token] if token in ast_w2i else ast_w2i['<UNK>'] for token in line[:config.MAX_SEQ_LEN]] for line in x_raw]
            y = [[nl_w2i[token] if token in nl_w2i else nl_w2i['<UNK>'] for token in line[:config.MAX_COMMENT_LEN]] for line in y_raw]
            if method == "TLCodeSum" or method == "RLComAPI":
                z = [[api_w2i[token] if token in api_w2i else api_w2i['<UNK>'] for token in line[:config.MAX_API_LEN]] for line in z_raw]                
                x = [x,z]
            if pretrain!="":
                loss = model.pretrain(x,y,pretrain)
            else:
                loss = model(x,True,y)
            logger.info("Epoch={}/{}, Batch={}, train_loss={}, time={}".format(epoch,config.EPOCH,has_trained,loss,get_time(time()-t0))) 
        
        bleu = []
        has_valid = 0
        valid_dataset = BackgroundGenerator(get_batch(valid_ast,valid_nl,config.EVAL_BATCH_SIZE,valid_api), 1)
        corpus = []
        targets = []
        for x_raw,y_raw,z_raw in valid_dataset:
            has_valid += 1
            x = [[ast_w2i[token] if token in ast_w2i else ast_w2i['<UNK>'] for token in line[:config.MAX_SEQ_LEN]] for line in x_raw]
            y = [[nl_w2i[token] if token in nl_w2i else nl_w2i['<UNK>'] for token in line[:config.MAX_COMMENT_LEN]] for line in y_raw]
            if method == "TLCodeSum" or method == "RLComAPI":
                z = [[api_w2i[token] if token in api_w2i else api_w2i['<UNK>'] for token in line[:config.MAX_API_LEN]] for line in z_raw]   
                x = [x,z]             
            preds = model(x,False)
            score = 0
            for i in range(len(preds)):
                y_ = [nl_i2w[token] if token in nl_i2w else "<UNK>" for token in y[i]]
                y_ = [token.lower() for token in y_]
                pred_ = [nl_i2w[token] if token in nl_i2w else "<UNK>" for token in preds[i]]
                pred_ = [token.lower() for token in pred_]
                score += score_sentence(pred_,y_,4,1)[-1]
                corpus += [pred_]
                targets += [y_]
            bleu.append(score/len(preds))
            logger.info("Epoch={}/{}, Batch={}, valid_sentence_bleu={}, time={}".format(epoch,config.EPOCH,has_valid,np.mean(bleu),get_time(time()-t0)))
                
        score = score_corpus(corpus,targets,4)
        logger.info("Epoch={}/{}, Batch={}, valid_corpus_bleu={}, time={}".format(epoch,config.EPOCH,has_valid,score,get_time(time()-t0)))

        score = np.mean(bleu)
        if score>best_bleu:
            best_bleu = score
            logger.info("Saving the model {} with the best bleu score {}\n".format(epoch, best_bleu))
            model.save(os.path.join(save_path,"{}_{}.pkl".format(datatype,epoch)))

        bleu = []
        has_test = 0
        test_dataset = BackgroundGenerator(get_batch(test_ast,test_nl,config.TEST_BATCH_SIZE,test_api), 1)
        corpus = []
        targets = []
        for x_raw,y_raw,z_raw in test_dataset:
            has_test += 1
            x = [[ast_w2i[token] if token in ast_w2i else ast_w2i['<UNK>'] for token in line[:config.MAX_SEQ_LEN]] for line in x_raw]
            y = [[nl_w2i[token] if token in nl_w2i else nl_w2i['<UNK>'] for token in line[:config.MAX_COMMENT_LEN]] for line in y_raw]
            if method == "TLCodeSum" or method == "RLComAPI":
                z = [[api_w2i[token] if token in api_w2i else api_w2i['<UNK>'] for token in line[:config.MAX_API_LEN]] for line in z_raw]     
                x = [x,z]           
            preds = model(x,False)
            score = 0
            for i in range(len(preds)):
                y_ = [nl_i2w[token] if token in nl_i2w else "<UNK>" for token in y[i]]
                y_ = [token.lower() for token in y_]
                pred_ = [nl_i2w[token] if token in nl_i2w else "<UNK>" for token in preds[i]]
                pred_ = [token.lower() for token in pred_]
                score += score_sentence(pred_,y_,4,1)[-1]
                corpus += [pred_]
                targets += [y_]
            bleu.append(score/len(preds))
            logger.info("Epoch={}/{}, Batch={}, test_sentence_bleu={}, time={}".format(epoch,config.EPOCH,has_test,np.mean(bleu),get_time(time()-t0)))
        
        score = score_corpus(corpus,targets,4)
        logger.info("Epoch={}/{}, Batch={}, test_corpus_bleu={}, time={}".format(epoch,config.EPOCH,has_test,score,get_time(time()-t0)))
