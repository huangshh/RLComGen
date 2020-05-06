import os

path = "./dataset/"
for mode in ["train","valid","test"]:
    token_path = os.path.join(path,"{}.token-nl2.token".format(mode))
    api_path = os.path.join(path,"{}.token-nl2.api".format(mode))
    with open(token_path,encoding="utf8") as f1, open(api_path,"w",encoding="utf8") as f2:
        for line in f1:
            ast_token = line.strip().split(" ")
            api_token = []
            num = 0
            for token in ast_token:
                p = True
                if token == "ClassInstanceCreation_Begin" or token == "MethodInvocation_Begin": 
                    num += 1
                    p = False
                if token == "ClassInstanceCreation_End" or token == "MethodInvocation_End": num -= 1
                if num > 1 or (num == 1 and p == True):
                    api_token.append("_".join(token.split("_")[1:]))
            f2.write(" ".join(api_token))
            f2.write("\n")