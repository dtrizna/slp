# NumPy: "The fundamental package for scientific computing with Python"
# allows to work with vectors, matrices, etc., aka Linear Algebra
import numpy as np

# packages needed to fetch data
import io
import re
import zipfile
import requests

# specifying a static random seed for reproducibility purposes
RANDOM_SEED = 1337

# benign command data samples
benign = [x.strip() for x in requests.get("https://raw.githubusercontent.com/dtrizna/slp/main/data/nl2bash.cm").text.strip().split("\n")]
# with open("/Users/dmitrijs.trizna/Desktop/slp/data/nl2bash.cm") as f:
#     benign = [x.strip() for x in f.readlines()]
auditd_execve_sample = requests.get("https://raw.githubusercontent.com/dtrizna/slp/main/data/exeve_sample.log").text.strip().split("\n")

# malicious command dataset
zraw = requests.get("https://github.com/dtrizna/slp/raw/main/data/malicious.zip").content
# with open("../data/malicious.zip", "rb") as f:
#     zraw = f.read()
with zipfile.ZipFile(io.BytesIO(zraw)) as z:
    with z.open("malicious.cm", pwd="infected".encode()) as f2:
        malicious = [x.strip().decode() for x in f2.readlines()]

# joining datasets together and assigning labels
X_raw = benign + malicious
y = np.array([0] * len(benign) + [1] * len(malicious), dtype=int)

from sklearn.feature_extraction.text import HashingVectorizer
from ..slp import ShellTokenizer

#aa,bb = ShellTokenizer(verbose=True, debug=False).tokenize(X_raw)

# works
hv = HashingVectorizer(
    lowercase=False,
    analyzer=ShellTokenizer(verbose=False, debug=True).tokenize_command
)
aa = hv.fit_transform(X_raw)

# works 2 -- w/ IP ADDRESS NORMALIZATION
hv2 = HashingVectorizer(
    lowercase=False,
    tokenizer=ShellTokenizer(verbose=False, debug=True).tokenize_command,
    # IP ADDRESS NORMALIZATION
    preprocessor=lambda x: re.sub(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", "IPADDRESS", x),
    token_pattern=None,
    max_features=1000
)

bb = hv2.fit_transform(X_raw)
import pdb;pdb.set_trace()