import sys
from os import path 

scriptpath = ".."
sys.path.append(path.abspath(scriptpath))
scriptpath = "."
sys.path.append(path.abspath(scriptpath))

from bashprocessing import Parser

def get_data(filepath):
    try:
        with open('../' + filepath) as f:
            data = f.readlines()
    except FileNotFoundError:
        with open(filepath) as f:
            data = f.readlines()
    return data

data = get_data('data/nl2bash.cm')
data = [x.strip() for x in data]

p = Parser(verbose=True)
counted, corpus = p.tokenize(data[0:300])

p = Parser(verbose=True)
tfidf = p.encode(mode='tf-idf', corpus=corpus[0:5], cntr=counted, top_tokens=10)
print(tfidf)
print(tfidf.toarray())
print(tfidf.shape)

oh = p.encode(mode='onehot', corpus=corpus[0:5], cntr=counted, top_tokens=10)
print(oh)
print(oh.shape)
print(type(oh))