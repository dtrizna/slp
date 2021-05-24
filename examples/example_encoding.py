import sys
from os import path 
from pprint import pprint

scriptpath = ".."
sys.path.append(path.abspath(scriptpath))
scriptpath = "."
sys.path.append(path.abspath(scriptpath))

from slp import ShellTokenizer, ShellEncoder

def get_data(filepath):
    try:
        with open('../' + filepath) as f:
            data = f.readlines()
    except FileNotFoundError:
        with open(filepath) as f:
            data = f.readlines()
    return data

data = get_data('data/nl2bash.cm')

t = ShellTokenizer(debug=True, verbose=True)
# take only first 100 commands for example purposes
corpus, counter = t.tokenize(data[0:100])

TOP = 500
encoder = ShellEncoder(corpus, counter, top_tokens=TOP, verbose=False)


print("\n", "Total unique elements found: ", len(counter))
print(counter.most_common(5))
"""
% python3 encoding_example.py
[!] Parsing in process.. 12606\12607
Total unique elements found: 10372
[('find', 7846), ('|', 6487), ('.', 3775), ('-name', 3616), ('-type', 3403)]
"""

# ENCODING EXAMPLES
labels = encoder.labels(pad_width=500)
# shape: (samples, padding_width)
print(labels.shape)
pprint(labels.toarray()[:5,:10])
"""
(100, 500)
array([[230,  34,  41,  63, 248, 220,  43,  84, 248, 220],
       [230,  34,  51,  82,  67, 128, 248, 133, 110, 227],
       [230,  34,  39,  93,  51,  90, 248, 133,  82,  99],
       [230,  51,  82,   0,   0,   0,   0,   0,   0,   0],
       [230,  36, 248, 165, 243,   0,   0,   0,   0,   0]])
"""

onehot = encoder.onehot()
# shape: (samples, top_tokens)
print(onehot.shape)
pprint(onehot.toarray()[:5,:])
"""
(100, 500)
array([[1, 1, 0, ..., 0, 0, 0],
       [1, 1, 1, ..., 0, 0, 0],
       [1, 1, 1, ..., 0, 0, 0],
       [0, 1, 0, ..., 0, 0, 0],
       [1, 1, 0, ..., 0, 0, 0]])
"""

tfidf = encoder.tfidf()
# shape: (samples, top_tokens)
print(tfidf.shape)
pprint(tfidf.toarray()[:5,:])
"""
(100, 500)
array([[0.15437351, 0.09073   , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.05145784, 0.06048667, 0.2277968 , ..., 0.        , 0.        ,
        0.        ],
       [0.03704964, 0.0435504 , 0.16401369, ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.36292   , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.18524821, 0.217752  , 0.        , ..., 0.        , 0.        ,
        0.        ]])
"""