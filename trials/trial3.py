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
num_heads = 1  # Using a single head for simplicity

# Token embedding matrix
token_embedding = nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding(torch.tensor(tokenized_sequence))

# Position embedding matrix
position_embedding = nn.Embedding(sequence_length, embedding_dim)
position_indices = torch.arange(sequence_length).unsqueeze(0)
position_embeddings = position_embedding(position_indices)

# Calculate input embedding matrix
input_embeddings = token_embeddings + position_embeddings.squeeze(0)

# Define Query, Key, and Value weight matrices
W_Q = nn.Linear(embedding_dim, 16, bias=False)
W_K = nn.Linear(embedding_dim, 16, bias=False)
W_V = nn.Linear(embedding_dim, 16, bias=False)

# Compute Query, Key, and Value matrices
Q = W_Q(input_embeddings)  # Shape (T, 16)
K = W_K(input_embeddings)  # Shape (T, 16)
V = W_V(input_embeddings)  # Shape (T, 16)

# Compute attention scores
d_k = Q.size(-1)  # Dimension of the Key vectors
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # Shape (T, T)

# Apply attention mask (all ones for simplicity)
attention_mask = torch.ones(sequence_length, sequence_length)
scores = scores.masked_fill(attention_mask == 0, float('-inf'))

# Compute attention weights
attention_weights = F.softmax(scores, dim=-1)

# Compute attention output
attention_output = torch.matmul(attention_weights, V)

# Output the results
print("Token Embeddings:\n", token_embeddings.shape)
print("Position Embeddings:\n", position_embeddings.shape)
print("Input Embeddings:\n", input_embeddings.shape)
print("Query Matrix (Q):\n", Q.shape)
print("Key Matrix (K):\n", K.shape)
print("Value Matrix (V):\n", V.shape)
print("Attention Scores:\n", scores.shape)
print("Attention Weights:\n", attention_weights.shape)
print("Attention Output:\n", attention_output)
