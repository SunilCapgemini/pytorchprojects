import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, sequence, target_sequence, embedding_dim=48, num_heads=3):
        super(TransformerModel, self).__init__()
        
        self.sequence = sequence
        self.target_sequence = target_sequence
        self.vocab = sorted(set(sequence))
        self.token_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_token = {idx: char for char, idx in self.token_to_index.items()}
        
        self.tokenized_sequence = [self.token_to_index[char] for char in sequence]
        self.tokenized_target_sequence = [self.token_to_index[char] for char in target_sequence]

        self.sequence_length = len(sequence)
        self.vocab_size = len(self.vocab)
        self.embedding_dim = embedding_dim  # Embedding size (C)
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embedding_dim // num_heads  # Dimension per head
        
        # Token embedding matrix
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Position embedding matrix
        self.position_embedding = nn.Embedding(self.sequence_length, self.embedding_dim)
        
        # Define the Transformer components
        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_O = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)  # Output projection
        
        # Define the MLP with one hidden layer using GeLU activation
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Define Layer Normalization layers
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)
        
        # Define the LM head
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def prepare_dataset(self, sequences, target_sequences):
        batch_tokenized_sequences = [torch.tensor([self.token_to_index[char] for char in seq]) for seq in sequences]
        batch_tokenized_target_sequences = [torch.tensor([self.token_to_index[char] for char in tgt_seq]) for tgt_seq in target_sequences]
        
        # Pad sequences to the same length
        padded_sequences = nn.utils.rnn.pad_sequence(batch_tokenized_sequences, batch_first=True)
        padded_target_sequences = nn.utils.rnn.pad_sequence(batch_tokenized_target_sequences, batch_first=True)
        
        token_embeddings = self.token_embedding(padded_sequences)

        position_indices = torch.arange(padded_sequences.size(1)).unsqueeze(0).expand(padded_sequences.size(0), -1)
        position_embeddings = self.position_embedding(position_indices)
        input_embeddings = token_embeddings + position_embeddings

        return input_embeddings, padded_target_sequences
    
    def forward(self, input_embeddings):
        input_embeddings_mean = input_embeddings.mean(dim=-1, keepdim=True)
        input_embeddings_std = input_embeddings.std(dim=-1, keepdim=True)
        ln_agg_input_embeddings = (input_embeddings - input_embeddings_mean) / (input_embeddings_std + 1e-6)
        ln_agg_input_embeddings = ln_agg_input_embeddings * input_embeddings_std + input_embeddings_mean

        normalized_input_embeddings = self.layer_norm1(ln_agg_input_embeddings)

        Q = self.W_Q(normalized_input_embeddings)
        K = self.W_K(normalized_input_embeddings)
        V = self.W_V(normalized_input_embeddings)

        print('Q K V shapes',Q.shape, K.shape, V.shape)

        Q = Q.view(3,self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(3,self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(3,self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        print('Q K V shapes',Q.shape, K.shape, V.shape)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_mask = torch.ones(self.sequence_length, self.sequence_length)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        print('attention_output',attention_output.shape)
        print('attention_output transpose',attention_output.transpose(1,2).shape)

        concatenated_output = attention_output.transpose(1, 2).contiguous().view(3, self.sequence_length, self.embedding_dim)
        exit()

        final_output = self.W_O(concatenated_output)
        mlp_output = self.mlp(concatenated_output)
        mlp_residuals = mlp_output + concatenated_output

        mlp_residuals_mean = mlp_residuals.mean(dim=-1, keepdim=True)
        mlp_residuals_std = mlp_residuals.std(dim=-1, keepdim=True)
        ln_agg_mlp_residuals = (mlp_residuals - mlp_residuals_mean) / (mlp_residuals_std + 1e-6)
        ln_agg_mlp_residuals = ln_agg_mlp_residuals * mlp_residuals_std + mlp_residuals_mean

        normalized_mlp_residuals = self.layer_norm2(ln_agg_mlp_residuals)
        logits = self.lm_head(normalized_mlp_residuals)

        return logits
    
    def train_model(self, num_epochs=1000,sequences=[],target_sequences=[]):
        for epoch in range(num_epochs):
            input_embeddings, padded_target_sequences = self.prepare_dataset(sequences, target_sequences)
            logits = self.forward(input_embeddings)
            loss = self.criterion(logits.view(-1, self.vocab_size), torch.tensor(padded_target_sequences))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def test(self,sequence):
        self.tokenized_sequence = [self.token_to_index[char] for char in sequence]
        input_embeddings = self.prepare_dataset()
        logits = self.forward(input_embeddings)
        logits_softmax = F.softmax(logits, dim=-1)
        predicted_indices = torch.argmax(logits_softmax, dim=-1)
        predicted_tokens = [self.index_to_token[idx.item()] for idx in predicted_indices]
        return predicted_tokens

    def validate(self):
        return self.test()

# Define the sequence and target sequence
sequence = "CBABBC"
target_sequence = "ABBBCC"

# Initialize the model
model = TransformerModel(sequence, target_sequence)

# Train the model
model.train_model(num_epochs=1000,sequences=['CABBAB','CAAABC','BAAABA'],target_sequences=['AABBBC','AAABBC','AAAABB'])
# Test the model
predicted_tokens = model.test('CABBAB')
print("Predicted Tokens:", predicted_tokens)
predicted_tokens = model.test('CAAABC')
print("Predicted Tokens:", predicted_tokens)
predicted_tokens = model.test('BAAABA')
print("Predicted Tokens:", predicted_tokens)
