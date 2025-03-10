import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the sequence
sequence = "CBABBC"

# Vocabulary and tokenization
vocab = sorted(set(sequence))
token_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_token = {idx: char for char, idx in token_to_index.items()}

# Tokenize the sequence
tokenized_sequence = [token_to_index[char] for char in sequence]

# Parameters
sequence_length = len(sequence)
vocab_size = len(vocab)
embedding_dim = 48  # Embedding size (C)
num_heads = 3  # Number of attention heads
head_dim = embedding_dim // num_heads  # Dimension per head

# Token embedding matrix
token_embedding = nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding(torch.tensor(tokenized_sequence))

# Position embedding matrix
position_embedding = nn.Embedding(sequence_length, embedding_dim)
position_indices = torch.arange(sequence_length).unsqueeze(0)
position_embeddings = position_embedding(position_indices)

# Calculate input embedding matrix
input_embeddings = token_embeddings + position_embeddings.squeeze(0)

# Apply LN aggregation to input_embeddings before Layer Normalization
input_embeddings_mean = input_embeddings.mean(dim=-1, keepdim=True)
input_embeddings_std = input_embeddings.std(dim=-1, keepdim=True)
ln_agg_input_embeddings = (input_embeddings - input_embeddings_mean) / (input_embeddings_std + 1e-6)
ln_agg_input_embeddings = ln_agg_input_embeddings * input_embeddings_std + input_embeddings_mean

# Apply Layer Normalization to the LN aggregated input embeddings
layer_norm1 = nn.LayerNorm(embedding_dim)
normalized_input_embeddings = layer_norm1(ln_agg_input_embeddings)

# Define Query, Key, and Value weight matrices
W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_O = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Output projection

# Compute Query, Key, and Value matrices
Q = W_Q(normalized_input_embeddings)  # Shape (T, C)
K = W_K(normalized_input_embeddings)  # Shape (T, C)
V = W_V(normalized_input_embeddings)  # Shape (T, C)
print('Q K V shapes', Q.shape)
print('Q K V shapes', K.shape)
print('Q K V shapes', V.shape)
# Split Q, K, V into multiple heads
Q = Q.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)
K = K.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)
V = V.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)

# Compute attention scores for each head
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))  # Shape (num_heads, T, T)

# Apply attention mask (all ones for simplicity)
attention_mask = torch.ones(sequence_length, sequence_length)
attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

# Compute attention weights for each head
attention_weights = F.softmax(attention_scores, dim=-1)  # Shape (num_heads, T, T)

# Compute attention output for each head
attention_output = torch.matmul(attention_weights, V)  # Shape (num_heads, T, head_dim)

# Concatenate the outputs of all heads
concatenated_output = attention_output.transpose(0, 1).contiguous().view(sequence_length, embedding_dim)  # Shape (T, C)

# Apply the final linear transformation
final_output = W_O(concatenated_output)  # Shape (T, C)

# Define the MLP with one hidden layer using GeLU activation
mlp = nn.Sequential(
    nn.Linear(embedding_dim, embedding_dim),
    nn.GELU(),
    nn.Linear(embedding_dim, embedding_dim)
)

# Apply the MLP to the concatenated output
mlp_output = mlp(concatenated_output)

# Calculate the residual connection
mlp_residuals = mlp_output + concatenated_output

# Apply LN aggregation to mlp_residuals before final Layer Normalization
mlp_residuals_mean = mlp_residuals.mean(dim=-1, keepdim=True)
mlp_residuals_std = mlp_residuals.std(dim=-1, keepdim=True)
ln_agg_mlp_residuals = (mlp_residuals - mlp_residuals_mean) / (mlp_residuals_std + 1e-6)
ln_agg_mlp_residuals = ln_agg_mlp_residuals * mlp_residuals_std + mlp_residuals_mean

# Apply final Layer Normalization
layer_norm2 = nn.LayerNorm(embedding_dim)
normalized_mlp_residuals = layer_norm2(ln_agg_mlp_residuals)

# Apply LM head weights to get logits
lm_head = nn.Linear(embedding_dim, vocab_size)
logits = lm_head(normalized_mlp_residuals)

# Calculate softmax of the logits
logits_softmax = F.softmax(logits, dim=-1)

# Output the results
print("Token Embeddings:\n", token_embeddings.shape)
print("Position Embeddings:\n", position_embeddings.shape)
print("Input Embeddings:\n", input_embeddings.shape)
print("LN Aggregated Input Embeddings:\n", ln_agg_input_embeddings.shape)
print("Normalized Input Embeddings:\n", normalized_input_embeddings.shape)
print("Query Matrix (Q):\n", Q.shape)
print("Key Matrix (K):\n", K.shape)
print("Value Matrix (V):\n", V.shape)
print("Attention Scores:\n", attention_scores.shape)
print("Attention Weights:\n", attention_weights.shape)
print("Attention Output (per head):\n", attention_output.shape)
print("Concatenated Output:\n", concatenated_output.shape)
print("Final Output:\n", final_output.shape)
print("MLP Residuals:\n", mlp_residuals.shape)
print("LN Aggregated MLP Residuals:\n", ln_agg_mlp_residuals.shape)
print("Normalized MLP Residuals:\n", normalized_mlp_residuals.shape)
print("Logits:\n", logits.shape)
print("Logits Softmax:\n", logits_softmax)

# Get the index with the highest probability for each token
predicted_indices = torch.argmax(logits_softmax, dim=-1)

# Convert the indices back to the corresponding vocabulary tokens
predicted_tokens = [index_to_token[idx.item()] for idx in predicted_indices]

print("Predicted Tokens:", predicted_tokens)

