import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the sequence and target sequence
sequence = "CBABBC"
target_sequence = "ABBBCC"

# Vocabulary and tokenization
vocab = sorted(set(sequence))
token_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_token = {idx: char for char, idx in token_to_index.items()}

# Tokenize the sequence and target sequence
tokenized_sequence = [token_to_index[char] for char in sequence]
tokenized_target_sequence = [token_to_index[char] for char in target_sequence]

# Parameters
sequence_length = len(sequence)
vocab_size = len(vocab)
embedding_dim = 48  # Embedding size (C)
num_heads = 3  # Number of attention heads
head_dim = embedding_dim // num_heads  # Dimension per head

# Token embedding matrix
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Position embedding matrix
position_embedding = nn.Embedding(sequence_length, embedding_dim)

# Define the Transformer components
W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_O = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Output projection

# Define the MLP linear layers
mlp_linear1 = nn.Linear(embedding_dim, embedding_dim)
mlp_linear2 = nn.Linear(embedding_dim, embedding_dim)

# Define Layer Normalization layers
layer_norm1 = nn.LayerNorm(embedding_dim)
layer_norm2 = nn.LayerNorm(embedding_dim)

# Define the LM head
lm_head = nn.Linear(embedding_dim, vocab_size)

# Optimizer and loss function
optimizer = optim.Adam(
    list(token_embedding.parameters()) +
    list(position_embedding.parameters()) +
    list(W_Q.parameters()) +
    list(W_K.parameters()) +
    list(W_V.parameters()) +
    list(W_O.parameters()) +
    list(mlp_linear1.parameters()) +
    list(mlp_linear2.parameters()) +
    list(layer_norm1.parameters()) +
    list(layer_norm2.parameters()) +
    list(lm_head.parameters()), lr=0.001
)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Tokenize the sequence
    tokenized_sequence_tensor = torch.tensor(tokenized_sequence)
    token_embeddings = token_embedding(tokenized_sequence_tensor)

    # Position embeddings
    position_indices = torch.arange(sequence_length).unsqueeze(0)
    position_embeddings = position_embedding(position_indices)

    # Input embeddings
    input_embeddings = token_embeddings + position_embeddings.squeeze(0)

    # --- start Transformer Block
    # Apply LN aggregation to input_embeddings before Layer Normalization
    input_embeddings_mean = input_embeddings.mean(dim=-1, keepdim=True)
    input_embeddings_std = input_embeddings.std(dim=-1, keepdim=True)
    ln_agg_input_embeddings = (input_embeddings - input_embeddings_mean) / (input_embeddings_std + 1e-6)
    ln_agg_input_embeddings = ln_agg_input_embeddings * input_embeddings_std + input_embeddings_mean

    # Apply Layer Normalization to the LN aggregated input embeddings
    normalized_input_embeddings = layer_norm1(ln_agg_input_embeddings)

    # Compute Query, Key, and Value matrices
    Q = W_Q(normalized_input_embeddings)  # Shape (T, C)
    K = W_K(normalized_input_embeddings)  # Shape (T, C)
    V = W_V(normalized_input_embeddings)  # Shape (T, C)

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
    concatenated_attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length, embedding_dim)  # Shape (T, C)

    # Apply the projection linear transformation
    final_output = W_O(concatenated_attention_output)  # Shape (T, C)

    # Calculate the attention residual
    attention_residual = final_output + input_embeddings

    # Apply LN aggregation to the attention residual
    attention_residual_mean = attention_residual.mean(dim=-1, keepdim=True)
    attention_residual_std = attention_residual.std(dim=-1, keepdim=True)
    ln_agg_attention_residual = (attention_residual - attention_residual_mean) / (attention_residual_std + 1e-6)
    ln_agg_attention_residual = ln_agg_attention_residual * attention_residual_std + attention_residual_mean

    # Apply Layer Normalization to the LN aggregated attention residual
    normalized_attention_residual = layer_norm2(ln_agg_attention_residual)

    # Apply MLP linear transformation
    mlp_output = mlp_linear1(normalized_attention_residual)

    # Apply GeLU activation
    mlp_activation = F.gelu(mlp_output)

    # Apply MLP projection weights
    mlp_projection_output = mlp_linear2(mlp_activation)

    # Get the MLP result
    mlp_residual = mlp_projection_output + normalized_attention_residual

    # Apply LN aggregation to the MLP residual before final Layer Normalization
    mlp_residual_mean = mlp_residual.mean(dim=-1, keepdim=True)
    mlp_residual_std = mlp_residual.std(dim=-1, keepdim=True)
    ln_agg_mlp_residual = (mlp_residual - mlp_residual_mean) / (mlp_residual_std + 1e-6)
    ln_agg_mlp_residual = ln_agg_mlp_residual * mlp_residual_std + mlp_residual_mean

    # Apply final Layer Normalization
    normalized_final_residual = layer_norm2(ln_agg_mlp_residual)
    #--- End Transformer block
    # Apply LM head weights to get logits
    logits = lm_head(normalized_final_residual)

    # Calculate softmax of the logits
    logits_softmax = F.softmax(logits, dim=-1)

    # Compute the loss
    loss = criterion(logits.view(-1, vocab_size), torch.tensor(tokenized_target_sequence))

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Output the results
predicted_indices = torch.argmax(logits_softmax, dim=-1)
predicted_tokens = [index_to_token[idx.item()] for idx in predicted_indices]
print("Predicted Tokens:", predicted_tokens)
