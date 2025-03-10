import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TransformerDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        return item

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

    def generate(self, input_ids, max_length=20):
        self.eval()
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# Hyperparameters
vocab_size = 30522  # Adjust according to your tokenizer vocabulary size
embed_dim = 128
num_heads = 8
hidden_dim = 512
num_layers = 2

# Sample data
texts = ["I love programming.", "Python is amazing.", "Transformers are powerful.", "AI is the future."]

# Tokenizer (using a simple tokenizer for this example)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

dataset = TransformerDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model, loss function, and optimizer
model = TransformerModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        labels = input_ids[1:]
        input_ids = input_ids[:-1]
        print(input_ids.shape)
        attention_mask = attention_mask[:-1].reshape(128,1)
        outputs = model(input_ids, attention_mask=attention_mask)
        print(outputs.shape)
        permuted_outputs = outputs.permute(0, 2, 1)
        print(permuted_outputs.shape)
        print('labels',labels.shape)
        loss = criterion(permuted_outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Test the model for sentence generation
def generate_sentence(text, max_length=20):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    input_ids = inputs['input_ids']
    generated_ids = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

test_text = "AI and"
print(generate_sentence(test_text))
