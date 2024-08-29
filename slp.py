import re
import bashlex
import logging
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


class ShellTokenizer():
    def __init__(self, debug=False, verbose=False):
        self.ERR = 0
        self.verbose = verbose
        self.data = None
        self.global_counter = Counter()
        self.tokenized_corpus = []

        # setup logging
        level = logging.INFO if verbose else logging.WARNING
        level = logging.DEBUG if debug else level
        logconfig = {
            "level": level,
            "format": "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
        }
        logging.basicConfig(**logconfig)

    def _bashlex_wrapper(self, command):
        try:
            nodes = bashlex.parse(command)
            return nodes
        # bashlex fails to parse some syntax combinations
        # in that case split command by <space>, <comma>, <curly_brackets>
        # and add values in between directly to counter as keys
        except (bashlex.errors.ParsingError, NotImplementedError, TypeError):
            self.ERR += 1
            rude_parse = []
            for element in re.split(r" |,|{|}", command):
                rude_parse.append(element)
            return rude_parse

    def _update_objects(self, counters, tokens):
        try:
            self.global_counter += counters
            self.tokenized_node.extend(tokens)
        except Exception as ex:
            print(ex)
            import pdb;pdb.set_trace()

    # self-calling iterative function
    def _iterate_bashlex(self, bashlex_object):
        local_counter = Counter()
        local_tokens = []

        # handling different bashlex object types into elements of Counter()
        if isinstance(bashlex_object, list):
            for element in bashlex_object:
                self._update_objects(*self._iterate_bashlex(element))
        
        elif isinstance(bashlex_object, bashlex.ast.node):
            object_methods = [x for x in dir(bashlex_object) if '__' not in x]
        
            if 'command' in object_methods:
                self._update_objects(*self._iterate_bashlex(bashlex_object.command))
            elif 'list' in object_methods:
                for element in bashlex_object.list:
                    self._update_objects(*self._iterate_bashlex(element))
            elif 'parts' in object_methods:        
                # if parts have parts within - go deeper 
                # (Leo from Inception looking at you)
                if bashlex_object.parts:
                    for part in bashlex_object.parts:
                        self._update_objects(*self._iterate_bashlex(part))
                else:
                    local_counter, local_tokens = self._parse_word(bashlex_object)
            elif 'word' in object_methods:
                local_counter, local_tokens = self._parse_word(bashlex_object)
            # Working on default specific types
            elif 'value' in object_methods:
                self.global_counter[bashlex_object.value] += 1
                self.tokenized_node.append(bashlex_object.value)
            elif "pipe" in object_methods:
                self.global_counter[bashlex_object.pipe] += 1
                self.tokenized_node.append(bashlex_object.pipe)
            elif "op" in object_methods:
                self.global_counter[bashlex_object.op] += 1
                self.tokenized_node.append(bashlex_object.op)
            elif "type" in object_methods:
                self.global_counter[bashlex_object.type] += 1
                self.tokenized_node.append(bashlex_object.type)
            else:
                logging.info(f"[DEBUG] Weird case - not parsed correctly: {bashlex_object}")
        
        return local_counter, local_tokens

    def _parse_word(self, bashlex_object):
        local_counter = Counter()
        local_tokens = []
        size = 1
        # Trying to parse input object again
        # helps if previous parse iteration didn't handled
        # some fragments of command (e.g. $(), ``, embedded ones)
        word = re.sub(r"[<>#{}]", "", bashlex_object.word).strip()
        if len(bashlex_object.word) > 20:
            p = self._bashlex_wrapper(word)
            if isinstance(p[0], bashlex.ast.node):
                try:
                    size = len(p[0].parts)
                except AttributeError:
                    size = len(p[0].list)
        # if size > 1 then parsing found something new
        if size > 1:
            for node in p:
                self._update_objects(*self._iterate_bashlex(node))
        # if no new elements - improve bashlex parsing of flags w/ '='
        elif '=' in bashlex_object.word and \
        '==' not in bashlex_object.word:
            l = bashlex_object.word.split('=')
            # special case
            if 'chmod' in bashlex_object.word.lower():
                local_counter[l[0]] += 1
                local_counter['='.join(l[1:])] += 1
                local_tokens.extend([l[0], '='.join(l[1:])])
            # standard 'flag=value' case
            elif len(l) == 2:
                local_counter[l[0]] += 1
                local_counter[l[1]] += 1
                local_tokens.extend([l[0], l[1]])
            # weird combination - add as is
            else:
                local_counter[bashlex_object.word] += 1
                local_tokens.append(bashlex_object.word)
        # bashlex parsing correct
        else:
            local_counter[bashlex_object.word] += 1
            local_tokens.append(bashlex_object.word)

        return local_counter, local_tokens
    
    def tokenize_corpus(self, corpus):
        l = len(corpus)
        for i, command in tqdm(enumerate(corpus), desc="[*] Tokenizing commands", total=l):
            self.tokenized_corpus.append(self.tokenize_command(command, i=i))
        return self.tokenized_corpus, self.global_counter
    
    def tokenize_command(self, command, i="*"):
        tokenized_command = []
        # bashlex wrapper returns list
        nodes = self._bashlex_wrapper(command)
        # if it consists of strings - then parsing failed 
        # and rude splitting is performed, add splitted command to corpus as is
        if isinstance(nodes[0], str):
            logging.debug(f"[{i}] 'bashlex' failed, regex tokenization:\n\t{command}\n\t{nodes}\n")
            
            tokenized_command.extend(nodes)
            self.global_counter += Counter(nodes)
        # if it consists of bashlex nodes - parsing suceeded
        elif isinstance(nodes[0], bashlex.ast.node):
            for node in nodes:
                self.tokenized_node = []
                self._update_objects(*self._iterate_bashlex(node))
                tokenized_command.extend(self.tokenized_node)
        else:
            logging.info(f"[-] Unexpected return type from 'bashlex', skipping command:\n\t{command}")
        
        return tokenized_command

    def tokenize(self, data):
        return self.tokenize_corpus(data)


class ShellEncoder():
    def __init__(self, corpus=None, token_counter=None, top_tokens=100, verbose=False):
        self.corpus = corpus
        self.token_counter = token_counter
        self.top_tokens = top_tokens
        self.verbose = verbose
        
        if not self.token_counter:
            self.token_counter = Counter([y for x in self.corpus for y in x])

        if not self.corpus or not self.token_counter:
            raise ValueError("[!] Please specify your corpus or use Parser().tokenize() to build it beforehand!")

        self.l = len(self.corpus)
        self.most_common = self.token_counter.most_common(self.top_tokens)
        self.top_token_list = [x[0] for x in self.most_common]


    def tfidf(self):
        # Inverse-Data-Frequency (IDF)
        idf = defaultdict(float)
        for token in self.top_token_list:
            appearance_in_corpus = 0
            for cmd in self.corpus:
                if token in cmd:
                    appearance_in_corpus += 1
            # 1 + is needed to not divide by 0 when there is no token in corpus
            idf[token] = np.log((1 + self.l) /(1 + appearance_in_corpus))

        # TF-IDF
        tfidf = np.zeros((self.l, self.top_tokens))
        logging.info(f"[!] Starting TF-IDF encoding!")
        for i,cmd in enumerate(self.corpus):
            omc = Counter(cmd)
            tf = defaultdict(float)            
            for token in set(cmd):
                # Term Frequency (TF) == Bag of Words (BoW)
                tf[token] = omc[token]/len(cmd)
                if token in list(idf):
                    idx = list(idf).index(token)
                    tfidf[i,idx] = tf[token] * idf[token]
                else:
                    # token not in top token list, omit
                    pass
        logging.info(f"[!] TF-IDF encoding finished!")
        return csr_matrix(tfidf)
    
    
    def labels(self, pad_width=None):          
        self.top_token_list = [x[0] for x in self.most_common]
        local_corpus = self.corpus.copy()

        le = LabelEncoder().fit(self.top_token_list+["OTHER"])
        for i,cmd in enumerate(self.corpus):
            local_corpus[i] = le.transform([x if x in self.top_token_list else "OTHER" for x in cmd])
        
        max_pad = pad_width if pad_width else np.max([len(x) for x in local_corpus])
        output = np.zeros((len(local_corpus), max_pad), dtype= np.int)
        
        logging.info(f"[!] Starting Label encoding!")
        for i,x in enumerate(local_corpus):
            z = np.zeros(max_pad)
            if max_pad > x.shape[0]:
                z[:x.shape[0]] = x
            else:
                z = x[:max_pad]
            output[i] = z
        logging.info(f"[!] Label encoding finished!")
        return csr_matrix(output)
        
    
    def onehot(self):
        output = np.zeros(shape=(self.l, self.top_tokens), dtype=int)
        logging.info(f"[!] Starting One-Hot encoding!")
        for i, cmd in enumerate(self.corpus):
            for j,token in enumerate(self.top_token_list):
                if token in cmd:
                    output[i,j] = 1
        logging.info(f"[!] One-Hot encoding finished!")
        return csr_matrix(output)
