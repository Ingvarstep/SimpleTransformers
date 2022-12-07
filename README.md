## Simple reconstruction of Bert and tokenizer for educational purposes

### Example:

```python
     bert = Bert(nhead, nlayer, dim_size, hid_dim, vocabsize, dropout)
     input_ids = torch.randint(0, vocabsize, (2, 10))
     atention_mask = torch.ones(2, 10)
     output = bert(input_ids, atention_mask)
```

where,
*   nhead - number of heads in multihead attention
*   nlayer - number of encoder layers
*   dim_size - dimension of embedding
*   hid_dim - dimension of hidden layer
*   vocabsize - size of vocabulary
*   dropout - dropout rate

