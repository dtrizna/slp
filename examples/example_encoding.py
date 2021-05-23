import sys
from os import path 
from pprint import pprint

scriptpath = ".."
sys.path.append(path.abspath(scriptpath))
scriptpath = "."
sys.path.append(path.abspath(scriptpath))

from slp import Parser

def get_data(filepath):
    try:
        with open('../' + filepath) as f:
            data = f.readlines()
    except FileNotFoundError:
        with open(filepath) as f:
            data = f.readlines()
    return data

data = get_data('data/nl2bash.cm')

p = Parser(verbose=True)

counted, corpus = p.tokenize(data[0:100])

print("\n", "Total unique elements found: ", len(counted))
print(counted.most_common(5))

"""
% python3 encoding_example.py
[!] Parsing in process.. 12606\12607
Total unique elements found: 10372
[('find', 7846), ('|', 6487), ('.', 3775), ('-name', 3616), ('-type', 3403)]
"""

# ENCODING EXAMPLES

labels = p.encode(mode="labels", top_tokens=10, pad_width=20)
# shape: (samples, padding_width)
print(labels.shape)
pprint(labels[:5,:10])
"""
(100, 20)
array([[ 9,  0,  4,  4, 10,  8,  4,  4, 10,  8],
       [ 9,  0,  1,  2,  4,  4, 10,  4,  4,  4],
       [ 9,  0,  4,  4,  1,  4, 10,  4,  2,  4],
       [ 9,  1,  2,  0,  0,  0,  0,  0,  0,  0],
       [ 9,  4, 10,  7,  4,  0,  0,  0,  0,  0]])
"""

onehot = p.encode(mode="onehot", top_tokens=10)
# shape: (samples, top_tokens)
print(onehot.shape)
pprint(onehot.toarray()[:5,:])
"""
(100, 10)
array([[1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
       [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
       [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
       [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 1, 0, 0]])
"""

tfidf = p.encode(mode="tf-idf", top_tokens=10)
# shape: (samples, top_tokens)
print(tfidf.shape)
pprint(tfidf.toarray()[:5,:])
"""
(100, 10)
array([[0.15437351, 0.09073   , 0.        , 0.        , 0.        ,
        0.        , 0.15892253, 0.        , 0.        , 0.38542257],
       [0.05145784, 0.06048667, 0.2277968 , 0.        , 0.10978129,
        0.11834521, 0.10594835, 0.        , 0.        , 0.        ],
       [0.03704964, 0.0435504 , 0.16401369, 0.        , 0.07904253,
        0.08520855, 0.07628281, 0.        , 0.        , 0.        ],
       [0.        , 0.36292   , 0.        , 0.        , 0.65868773,
        0.71007129, 0.        , 0.        , 0.        , 0.        ],
       [0.18524821, 0.217752  , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.41003423, 0.        , 0.        ]])
"""