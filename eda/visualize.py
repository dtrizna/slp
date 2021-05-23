import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import string
import sys

from collections import Counter
from os import path, getcwd

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

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

nl2bash = get_data('data/nl2bash.cm')

p = Parser(verbose=True)
counter, corpus = p.tokenize(nl2bash)

bash_stopwords = [key for key in counter.keys() if all(key.isdigit() or len(key) < 3 or j in string.punctuation for j in key)]
clean_dict = {key:counter[key] for key in counter.keys() if not all(key.isdigit() or len(key) < 3 or j in string.punctuation for j in key)}


# WORDCLOUD

# Read the image, prepare the wordcloud and image
d = path.dirname(__file__) if "__file__" in locals() else getcwd()
tux_coloring = np.array(Image.open(path.join(d, "../img/Tux_mask.jpg")))

image_colors = ImageColorGenerator(tux_coloring)
wc = WordCloud(background_color="white", max_words=200, mask=tux_coloring,
               stopwords=bash_stopwords)#, max_font_size=40)

# generate word cloud
wc.generate_from_frequencies(dict(clean_dict))
# recolor the penguin and show the image
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis('off')

#savefile = "img/Tux_wordcloud.png"
#plt.savefig(savefile, bbox_inches="tight")
# For some reason .savefig() provides weak quality...
# When saved manually quality is great!

plt.show()


# BARCHART

X,Y = zip(*Counter(clean_dict).most_common(12))

sns.set_theme(style="whitegrid")
sns.set(font_scale=1.4)
plt.figure(figsize=(14,6))
bars = sns.barplot(x=list(X), y=list(Y), palette="Blues_d")
#bars.get_figure().savefig("../img/absolute_element_counts.png")
plt.show()
#plt.close()

# BARCHART MALICIOUS

mal = get_data('data/malicious.cm')

p = Parser(verbose=True)
counter, corpus = p.tokenize(mal)

bash_stopwords = [key for key in counter.keys() if all(key.isdigit() or len(key) < 3 or j in string.punctuation for j in key)]
clean_dict = {key:counter[key] for key in counter.keys() if not all(key.isdigit() or len(key) < 3 or j in string.punctuation for j in key)}


X,Y = zip(*Counter(clean_dict).most_common(12))

sns.set_theme(style="whitegrid")
sns.set(font_scale=1.4)
plt.figure(figsize=(14,6))
bars = sns.barplot(x=list(X), y=list(Y), palette="rocket")
#bars.get_figure().savefig("../img/absolute_element_counts.png")
plt.show()
#plt.close()
