import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGNewsDataset(Dataset):
    def __init__(self, data, tokenizer, vocab=None, max_length=100):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        # Create a vocabulary based on word frequency
        counter = Counter()
        for text in self.data['Description']:
            tokens = self.tokenizer(text)
            counter.update(tokens)
        vocab = {word: idx + 1 for idx, (word, freq) in enumerate(counter.items())}
        vocab['<PAD>'] = 0
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Description']
        label = self.data.iloc[idx]['Class Index'] - 1  # 0-indexed
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] if token in self.vocab else 0 for token in tokens]
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)


tokenizer = word_tokenize 

train_dataset = AGNewsDataset(train_data, tokenizer)
val_dataset = AGNewsDataset(val_data, tokenizer, vocab=train_dataset.vocab)
test_dataset = AGNewsDataset(test_df, tokenizer, vocab=train_dataset.vocab)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = torch.mean(embedded, dim=1)
        return self.fc(pooled)

vocab_size = len(train_dataset.vocab)
embed_dim = 64
num_class = len(train_df['Class Index'].unique())

model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {correct_predictions/len(train_dataset):.4f}, Val Acc: {val_acc:.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    correct_predictions = 0
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    return correct_predictions / len(data_loader.dataset)

train_model(model, train_loader, val_loader, epochs=10)

test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "topic_classification_model.pth")

