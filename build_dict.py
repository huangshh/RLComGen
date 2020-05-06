import os
from collections import defaultdict, Counter
import pickle

path = "./dataset/"
token_path = os.path.join(path,"train.token-nl2.token")
code_path = os.path.join(path,"train.token-nl2.code2")
nl_path = os.path.join(path,"train.token-nl2.nl2")
api_path = os.path.join(path,"train.token-nl2.api")
ast_dict = defaultdict(int)
nl_dict = defaultdict(int)
api_dict = defaultdict(int)
code_dict = defaultdict(int)

with open(token_path,encoding="utf8") as f1, open(nl_path,encoding="utf8") as f2, open(api_path,encoding="utf8") as f3, open(code_path,encoding='utf8') as f4:
    for line1, line2, line3,line4 in zip(f1,f2,f3,f4):
        ast_token = line1.split(" ")
        nl_token = line2.split(" ")
        api_token = line3.split(" ")
        code_token = line4.split(" ")
        
        for token in ast_token:
            ast_dict[token] += 1
        for token in nl_token:
            nl_dict[token] += 1
        for token in api_token:
            api_dict[token] += 1
        for token in code_token:
            code_dict[token] += 1

dict_size = 50000
ast_dict = [tu[0] for tu in Counter(ast_dict).most_common(dict_size)]
nl_dict = [tu[0] for tu in Counter(nl_dict).most_common(dict_size)]
api_dict = [tu[0] for tu in Counter(api_dict).most_common(dict_size)]
code_dict = [tu[0] for tu in Counter(code_dict).most_common(dict_size)]
ast_dict = ["<START>","<END>","<UNK>","<PAD>"] + ast_dict
nl_dict = ["<START>","<END>","<UNK>","<PAD>"] + nl_dict
api_dict = ["<START>","<END>","<UNK>","<PAD>"] + api_dict
code_dict = ["<START>","<END>","<UNK>","<PAD>"] + code_dict

ast_w2i = {y:x for x,y in enumerate(ast_dict)}
ast_i2w = {x:y for x,y in enumerate(ast_dict)}
nl_w2i = {y:x for x,y in enumerate(nl_dict)}
nl_i2w = {x:y for x,y in enumerate(nl_dict)}
api_w2i = {y:x for x,y in enumerate(api_dict)}
api_i2w = {x:y for x,y in enumerate(api_dict)}
code_w2i = {y:x for x,y in enumerate(code_dict)}
code_i2w = {x:y for x,y in enumerate(code_dict)}

with open(os.path.join(path,"ast_w2i.pkl"),"wb") as f:
    pickle.dump(ast_w2i,f)
with open(os.path.join(path,"ast_i2w.pkl"),"wb") as f:
    pickle.dump(ast_i2w,f)
with open(os.path.join(path,"nl_w2i.pkl"),"wb") as f:
    pickle.dump(nl_w2i,f)
with open(os.path.join(path,"nl_i2w.pkl"),"wb") as f:
    pickle.dump(nl_i2w,f)
with open(os.path.join(path,"api_w2i.pkl"),"wb") as f:
    pickle.dump(api_w2i,f)
with open(os.path.join(path,"api_i2w.pkl"),"wb") as f:
    pickle.dump(api_i2w,f)
with open(os.path.join(path,"code_w2i.pkl"),"wb") as f:
    pickle.dump(code_w2i,f)
with open(os.path.join(path,"code_i2w.pkl"),"wb") as f:
    pickle.dump(code_i2w,f)

print(len(ast_w2i),len(nl_w2i),len(api_w2i),len(code_w2i))