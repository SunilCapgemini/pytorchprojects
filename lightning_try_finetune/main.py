import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        return tokens.input_ids[0], tokens.attention_mask[0]

class LLaMAModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    return input_ids, attention_mask

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
texts = dataset["train"]["text"]

# Initialize tokenizer and model
model_name = "EleutherAI/gpt-neox-20b"  # Replace with your LLaMA model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = TextDataset(texts, tokenizer)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = LLaMAModel(model_name=model_name)

# Initialize trainer
trainer = pl.Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)

# Train the model
trainer.fit(model, dataloader)

