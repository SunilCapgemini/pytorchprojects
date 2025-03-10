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

# Apply Layer Normalization to the input embeddings
layer_norm = nn.LayerNorm(embedding_dim)
normalized_input_embeddings = layer_norm(input_embeddings)

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

# Output the results
print("Token Embeddings:\n", token_embeddings.shape)
print("Position Embeddings:\n", position_embeddings.shape)
print("Input Embeddings:\n", input_embeddings.shape)
print("Normalized Input Embeddings:\n", normalized_input_embeddings.shape)
print("Query Matrix (Q):\n", Q.shape)
print("Key Matrix (K):\n", K.shape)
print("Value Matrix (V):\n", V.shape)
print("Attention Scores:\n", attention_scores.shape)
print("Attention Weights:\n", attention_weights.shape)
print("Attention Output (per head):\n", attention_output.shape)
print("Concatenated Output:\n", concatenated_output.shape)
print("Final Output:\n", final_output.shape)
