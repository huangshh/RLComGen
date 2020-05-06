from bleu import *

preds = []
filename = "pred.txt"
with open(filename) as f:
    for line in f:
        pred = line.split(" ")
        preds.append(pred)

targets = []
filename = "target.txt"
with open(filename) as f:
    for line in f:
        target = line.split(" ")
        targets.append(target)

corpus_bleu = score_corpus(preds,targets,4)
sentence_bleu = 0
for x,y in zip(preds,targets):
    sentence_bleu += score_sentence(x,y,4,1)[-1]

print('corpus_bleu: {}'.format(corpus_bleu))
print('sentence_bleu: {}'.format(sentence_bleu))