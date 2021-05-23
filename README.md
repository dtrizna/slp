# Shell Language Processing (SLP)

<p align="center"><img src="img/Tux_wordcloud.png" alt="WordCloud of most common elements" width="400"/>

## Example usage

```python
from slp import Parser

with open("all.cm") as file:
    data = file.readlines()

p = Parser()
data_counter, data_corpus = p.parse(data)
print(data_counter.most_common(5))

[Out]:
    [('find', 7846),
    ('|', 6487),
    ('.', 3775),
    ('-name', 3616),
    ('-type', 3403)]

encoded_X = p.encode(mode="tf-idf", corpus=data_corpus[0:5], token_counter=data_counter, top_tokens=10)
print(encoded_X.shape)
print(type(encoded_X))
print(encoded_X[0].toarray())

[Out]:
    (5,10)
    <class 'scipy.sparse.csr.csr_matrix'>
    [0.01012898         0.         0.         0.07701635        0.
    0.         0.03646431         0.         0.03038693         0.05545177]

```

At this point data is ready to be supplied as input for your ML model:

```python
mymodel.fit(encoded_X,y)
```

## Notes

- Pre-processing depends on `bashlex` (https://github.com/idank/bashlex) library.  
- Benign example dataset (i.e. `data/nl2bash.cm`) is based on `nl2bash` paper (https://arxiv.org/abs/1802.08979; https://github.com/TellinaTool/nl2bash).
- Malicious example dataset is collected from various Penetration Testing resources and scripts, some examples:
    - [Reverse Shell Cheat Sheet](https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Reverse%20Shell%20Cheatsheet.md)
    - [LinEnum.sh Script](https://github.com/rebootuser/LinEnum/blob/master/LinEnum.sh)
    - [Linux Privilege Escalation Guide](https://blog.g0tmi1k.com/2011/08/basic-linux-privilege-escalation/)

> Note that datasets are normalized so remote host is specified as `example.com`in both command sets.

- Some ideas of exploratory data analysis and visualiations can be found under `/eda/`:

<img src="img/absolute_element_counts.png" alt="Absolute Element Counts" width="600">
