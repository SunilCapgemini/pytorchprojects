import torch
import torch.nn as nn

# Define the sequence
sequence = "CBABBC"

# Vocabulary and tokenization
vocab = sorted(set(sequence))
token_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_token = {idx: char for char, idx in token_to_index.items()}
print(token_to_index)

# Tokenize the sequence
tokenized_sequence = [token_to_index[char] for char in sequence]
print(tokenized_sequence)
print(vocab)
# Parameters
sequence_length = len(sequence)
vocab_size = len(vocab)
embedding_dim = 8  # You can adjust the embedding dimension as needed

# Token embedding matrix
token_embedding = nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding(torch.tensor(tokenized_sequence))


# Position embedding matrix
position_embedding = nn.Embedding(sequence_length, embedding_dim)
position_indices = torch.arange(sequence_length).unsqueeze(0)
position_embeddings = position_embedding(position_indices)

# Calculate input embedding matrix
input_embeddings = token_embeddings + position_embeddings.squeeze(0)

# Output the results
print("Token Embeddings:\n", token_embeddings)
print("Token Embeddings:\n", token_embeddings.shape)
print("Position Embeddings:\n", position_embeddings)
print("Position Embeddings:\n", position_embeddings.shape)
print("Input Embeddings:\n", input_embeddings.shape)
