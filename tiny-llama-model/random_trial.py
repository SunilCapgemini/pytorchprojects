import torch
import torch.nn as nn

# Assume input_ids have shape (batch_size, sequence_length)
input_ids = torch.randint(0, 1000, (1, 128))  # Example input_ids tensor

# Correct src_key_padding_mask shape should be (batch_size, sequence_length)
attention_mask = torch.rand(1, 128) > 0.5  # Example mask with boolean values

print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)

# exit()
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.float()
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.fc(x)
        return x

# Example usage
model = TransformerModel(vocab_size=30522, embed_dim=128, num_heads=8, hidden_dim=512, num_layers=2)
output = model(input_ids, attention_mask=attention_mask)
print("Model output shape:", output.shape)
