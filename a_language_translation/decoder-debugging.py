import torch
import torch.nn as nn

encoder_outputs = torch.ones((32,10,128),dtype=torch.long)
encoder_hidden = torch.ones((1,32,128),dtype=torch.long)

batch_size = encoder_outputs.size(0) # = 32

decoder_input = torch.empty(32,1,dtype=torch.long).fill_(0)
print(decoder_input.shape)
decoder_hidden = encoder_hidden

decoder_outputs = []

for i in range(10): # Max-length = 10 of sentence
    decoder_output, decoder_hidden = forward_step(decoder_input)