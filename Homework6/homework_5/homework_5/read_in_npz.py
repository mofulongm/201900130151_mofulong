import numpy as np
a = np.load("C://Users//15339//Desktop//深度学习//实验//实验七//homework_5//homework_5//char-rnn-snapshot.npz",allow_pickle=True)
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()
