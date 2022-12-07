import torch
import math

class Bert(torch.nn.Module):
    def __init__(self, nhead, nlayer, dim_size, hid_dim, vocabsize, dropout):
        super(Bert, self).__init__()
        self.nhead = nhead
        self.nlayer = nlayer
        self.dim_size = dim_size
        self.hid_dim = hid_dim
        self.vocabsize = vocabsize
        self.dropout = dropout
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(nhead, dim_size, hid_dim, dropout) for _ in range(nlayer)])
        self.norm = torch.nn.LayerNorm(dim_size)
        self.embeddings = BertEmbeddings(vocabsize, dim_size, dropout)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)
        embeddings = self.norm(embeddings)
        return embeddings

class EncoderLayer(torch.nn.Module):
    def __init__(self, nhead, dim_size, hid_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.nhead = nhead
        self.dim_size = dim_size
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.self_attn = BertSelfAttention(nhead, dim_size, dropout)
        self.interlayer = InterLayer(dim_size, hid_dim, dropout)
        self.norm1 = torch.nn.LayerNorm(dim_size)

    def forward(self, embeddings, attention_mask):
        attn = self.self_attn(embeddings, attention_mask)
        out = self.interlayer(attn)
        out = self.norm1(out)
        return out

class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocabsize, dim_size, dropout):
        super(BertEmbeddings, self).__init__()
        self.vocabsize = vocabsize
        self.dim_size = dim_size
        self.dropout = dropout
        self.word_embeddings = torch.nn.Embedding(vocabsize, dim_size)

    def forward(self, input_ids):
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self._position_embeddings(input_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = torch.nn.functional.dropout(embeddings, self.dropout, self.training)
        return embeddings

    def _position_embeddings(self, input_ids):
        positions = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids) #(2, 10)
        #add a new dimension with dim_size
        positions = positions.unsqueeze(-1) #(2, 10, 1)
        #expand the new dimension with dim_size
        positions = positions.expand(-1, -1, self.dim_size) #(2, 10, dim_size)
        position_embeddings = torch.sin(positions / (10000 ** (torch.arange(self.dim_size, dtype=torch.float, device=input_ids.device) / self.dim_size)))
        return position_embeddings

class BertSelfAttention(torch.nn.Module):
    def __init__(self, nhead, dim_size, dropout):
        super(BertSelfAttention, self).__init__()
        self.nhead = nhead
        self.dim_size = dim_size
        self.dropout = dropout
        self.qkv = torch.nn.Linear(dim_size, dim_size * 3)
        self.out = torch.nn.Linear(dim_size, dim_size)
        self.act = torch.nn.ReLU()
        self.mh_attn = BertMultiHeadAttention(nhead, dim_size, dropout)

    def forward(self, input, attention_mask):
        qkv = self.qkv(input).chunk(3, dim=-1)
        q, k, v = [self._reshape(x) for x in qkv]
        attn = self.mh_attn(q, k, v, attention_mask)
        y = self.out(self.act(attn))
        return attn
    
    def _reshape(self, x):
        return x.view(x.size(0), x.size(1), self.nhead, self.dim_size // self.nhead).transpose(1, 2)


class BertMultiHeadAttention(torch.nn.Module):
    def __init__(self, nhead, dim_size, dropout):
        super(BertMultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.dim_size = dim_size
        self.dropout = dropout

    def forward(self, q, k, v, attention_mask):
        assert self.dim_size%self.nhead == 0
        dim_per_head = self.dim_size // self.nhead
        k = k.transpose(-1, -2)
        attn = torch.matmul(q, k)
        attn = attn / math.sqrt(dim_per_head)
        attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = torch.matmul(attn, v)
        attn = attn.transpose(1, 2).contiguous().view(attn.size(0), -1, self.dim_size)
        return attn

class InterLayer(torch.nn.Module):
    def __init__(self, dim_size, hid_dim, dropout):
        super(InterLayer, self).__init__()
        self.dim_size = dim_size
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(dim_size, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim, dim_size)
        self.act =  torch.nn.ReLU()
    
    def forward(self, input):
        x = self.linear1(input)
        x = self.act(x)
        x = torch.nn.functional.dropout(x, self.dropout, self.training)
        x = self.linear2(x)
        return x

