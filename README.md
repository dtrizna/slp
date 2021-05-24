# Shell Language Processing (SLP)

<p align="center"><img src="img/Tux_wordcloud.png" alt="WordCloud of most common elements" width="300"/>

## Example usage

```python
from slp import ShellTokenizer, ShellEncoder

with open("commands.txt") as file:
    data = file.readlines()

tokenizer = ShellTokenizer()
command_corpus, command_counter = tokenizer.tokenize(data)
print(command_counter.most_common(5))
"""
[('find', 7846),
('|', 6487),
('.', 3775),
('-name', 3616),
('-type', 3403)]
"""

encoder = ShellEncoder(command_corpus, command_counter, top_tokens=500, verbose=False)
X_tfidf = encoder.tfidf()
# shape: (commands, top_tokens)
print(X_tfidf.shape)
pprint(X_tfidf.toarray()[:5,:])
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
```

At this point data is ready to be supplied as input for your ML model:

```python
mymodel.fit(X_tfidf, y)
```

## Notes

- Pre-processing depends on [bashlex](https://github.com/idank/bashlex) library.  
- Benign example dataset (i.e. `data/nl2bash.cm`) is based on [nl2bash paper](https://arxiv.org/abs/1802.08979). Original nl2bash dataset can be found [here](https://github.com/TellinaTool/nl2bash).
- Malicious example dataset is collected from various Penetration Testing resources and scripts, some examples:
  - [Reverse Shell Cheat Sheet](https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Reverse%20Shell%20Cheatsheet.md)
  - [LinEnum.sh Script](https://github.com/rebootuser/LinEnum/blob/master/LinEnum.sh)
  - [Linux Privilege Escalation Guide](https://blog.g0tmi1k.com/2011/08/basic-linux-privilege-escalation/)

> Note that datasets are normalized so remote host is specified as `example.com`in both command sets.

- Some ideas of exploratory data analysis and visualiations can be found under `/eda/` and example code under `/examples/`:

<img src="img/roc_tfidf.png" alt="ROC curve for Cross-Validation of TF-IDF encoded data" width="700">
