import re
from tqdm import tqdm
import json

class Tokenizer:
    def __init__(self, special_tokens = ['<PAD>', '<UNK>'], count = 5000):
        self.tokens = []
        self.special_tokens = special_tokens
        self.count = count
        self.token2count = None

    def train(self, corpus):
        token2count = {}
        for text in tqdm(corpus):
            words = re.findall(r'\w+', text.lower())
            for word in words:
                if len(word)==1:
                    if word in token2count:
                        token2count[word] += 1
                    else:
                        token2count[word] = 1
                    continue
                for i in range(len(word)):
                    for j in range(i, len(word)):
                        token = word[i:j+1]
                        if token not in token2count:
                            token2count[token] = 1
                        else:
                            token2count[token] += 1
        self.tokens = set(dict(sorted(token2count.items(), key=lambda x: x[1], reverse=True)[:self.count]).keys())
        self.tokens.update(set(self.special_tokens))
        token2count.update({token: 0 for token in self.special_tokens})
        self.tokens = set(self.tokens)
        self.token2count = token2count

    def tokenize(self, text): 
        words = re.findall(r'\w+', text.lower())
        all_tokens = []
        for word in words:
            tokens = []
            combs = []
            if len(word)==1:
                if word in self.tokens:
                    all_tokens.append(word)
                else:
                    all_tokens.append('<UNK>')
                continue
            for i in range(len(word)):
                for j in range(i, len(word)):
                    token = word[i:j+1]
                    if token not in self.tokens:# or token=='##' or token=='####':
                        token = '<UNK>'
                    tokens.append(token)
                    combs.append((i, j))

            paths = []

            for i in range(len(combs)):
                path = []
                if combs[i][0] == 0:
                    comb = combs[i]
                    path.append(i)
                    for j in range(len(combs)):
                        if combs[j][0]==0:
                            continue
                        combj = combs[j]
                        if combj[0] == comb[-1]+1:
                            path.append(j)
                            comb = combj
                    paths.append(path)
            path2score = [[sum([self.token2count[tokens[i]] for i in path]), path] for path in paths]
            if len(path2score) > 0:
                paths = sorted(path2score, key=lambda x: x[0], reverse=True)
                path2len = [[len(path[1]), path] for path in paths]
                path2len = sorted(path2len, key=lambda x: x[0], reverse=False)
                path = path2len[0][1][1]
            else:
                all_tokens+=['<UNK>']
                continue
            all_tokens += [tokens[i] for i in path]

        return all_tokens

    @classmethod
    def from_file(self, file):
        with open(file, 'r') as f:
            token2count = json.load(f)
        tokens = set(token2count.keys())
        tokenizer = Tokenizer()
        tokenizer.tokens = tokens
        tokenizer.token2count = token2count
        return tokenizer
            
    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.token2count, f)

    
        

