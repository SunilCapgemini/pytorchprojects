import torch
import torch.nn.functional as F

# Example input sequence and attention mask
batch_size = 2
sequence_length = 5
d_model = 4

# Simulated input tensor (batch_size, sequence_length, d_model)
input_tensor = torch.randn(batch_size, sequence_length, d_model)

# Simulated attention mask (batch_size, sequence_length)
attention_mask = torch.tensor([[1, 1, 1, 0, 0],
                               [1, 1, 0, 0, 0]], dtype=torch.float32)

# Reshape attention mask to (batch_size, sequence_length, 1)
expanded_mask = attention_mask.unsqueeze(-1)
print(attention_mask.shape)
print(expanded_mask.shape)
# Compute attention scores (dot product of queries and keys)
queries = input_tensor
keys = input_tensor
scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
print("scores",scores.shape)
# Apply the attention mask (broadcasted to match scores shape)
scores = scores.masked_fill(expanded_mask == 0, float('-inf'))

# Apply softmax to get attention weights
attention_weights = F.softmax(scores, dim=-1)

print("Attention Weights:", attention_weights.shape)
