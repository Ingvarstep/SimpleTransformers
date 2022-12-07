from bert import Tokenizer, Bert
import torch

def test_tokenizer():
    tokenizer = Tokenizer()
    with open('data/data.txt', 'r') as f:
        corpus = f.read().splitlines()
    tokenizer.train(corpus)
    tokenizer.save('data/tokenizer.json')
    tokenizer = Tokenizer.from_file('data/tokenizer.json')
    tokens = tokenizer.tokenize('Hello, my name is John and I love biotechnadhjsd')
                        
def test_transformer():
     nhead = 4
     nlayer = 2
     dim_size = 16
     hid_dim = 32
     vocabsize = 100 
     dropout = 0.1
     
     bert = Bert(nhead, nlayer, dim_size, hid_dim, vocabsize, dropout)
     input_ids = torch.randint(0, vocabsize, (2, 10))
     atention_mask = torch.ones(2, 10)
     output = bert(input_ids, atention_mask)
