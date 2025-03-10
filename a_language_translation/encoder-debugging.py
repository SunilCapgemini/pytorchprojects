import torch.nn as nn
import torch

tr_embedding = nn.Embedding(2980,128)
tr_gru = nn.GRU(128,128)
tr_gruWithBatchfirst = nn.GRU(128,128,batch_first=True)

e_value = tr_embedding(torch.ones((32,10),dtype=torch.long))
drop_value = nn.Dropout(0.1)(e_value)
gru_value_output, gru_value_hidden = tr_gru(drop_value)
gru_value_output_bat, gru_value_hidden_bat = tr_gruWithBatchfirst(drop_value)
print(e_value.shape)
print(drop_value.shape)
print(gru_value_output.shape,gru_value_hidden.shape)
print(gru_value_output_bat.shape,gru_value_hidden_bat.shape)
print(gru_value_output_bat.size(0))
batch_size = gru_value_output_bat.size(0)
# = 32
encoder_outputs = gru_value_output_bat
# shape (32,10,128)
encoder_hidden = gru_value_hidden_bat
# shape (1,32,128)