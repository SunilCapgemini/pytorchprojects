import torch
import torch.nn as nn

# Define the TransformerEncoderLayer
d_model = 128  # Dimensionality of the input
nhead = 8  # Number of attention heads
dim_feedforward = 2048  # Dimensionality of the feedforward network
dropout = 0.1  # Dropout rate

encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# Example input with max sequence length of 128
batch_size = 1
seq_length = 128
input_tensor = torch.rand(batch_size, seq_length, d_model)

# Forward pass through the encoder layer
output = encoder_layer(input_tensor)
print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
mask = torch.ones((128,1))
print('mask.shape',mask)
enc_out = transformer_encoder(input_tensor,src_key_padding_mask=mask)
print("Enc Output shape:", enc_out.shape)
