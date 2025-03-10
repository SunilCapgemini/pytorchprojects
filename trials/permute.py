import torch.nn as nn
import torch
# This line of code creates a transformer encoder layer, which is a component of the
# transformer architecture. The layer takes in a tensor of shape (batch_size, seq_len, d_model)
# and outputs a tensor of the same shape. The layer consists of a multi-head self-attention
# mechanism, followed by a feedforward network (FFN). The multi-head self-attention mechanism
# is used to allow the model to attend to different parts of the input sequence simultaneously.
# The FFN is used to transform the output of the self-attention mechanism. The layer is
# configured to use the "batch_first" argument, which means that the batch dimension is the
# first dimension of the input tensor. The other arguments are the model dimension (d_model),
# the number of heads (nhead), and the dropout probability (dropout). The dropout probability
# is set to 0.1, which means that the layer will randomly drop out 10% of the input elements
# during training.
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, dropout=0.1)
src = torch.rand(32, 10, 512)
out = encoder_layer(src)
print(out.shape)