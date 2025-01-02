from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from .dataset import Dataset
from .model import EmotionClassifier
from .enums import EmotionType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from math import inf
import pandas as pd
import torch
import os

LEARNING_RATE = 0.00001
DATASET_FILE = "dataset.xlsx"

batch_size = int(input("Enter a batch size: "))
epochs = int(input("Number of epochs: "))
print("Loading dataset...")

df = pd.read_excel(DATASET_FILE)
texts = df['text'].tolist()
labels = df['classification'].tolist()
assert all(label in list(map(int, EmotionType)) for label in labels)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=52)
train_dataset = Dataset(train_texts, train_labels)
val_dataset = Dataset(val_texts, val_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

print("Dataset loaded!")

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA is available! Using it...")
else:
    print("CUDA isn't available, using CPU...")
device = torch.device("cuda" if use_cuda else "cpu")
model = EmotionClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_weights, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.CrossEntropyLoss(weights)
writer = SummaryWriter()

if os.path.isfile('model.pth'):
    question = input("Script found an existing model, do you want to train it? (y/n): ")
    question = question.lower()
    if question in ('y', 'yes'):
        model = model.load_state_dict(torch.load('model.pth'))


def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    loss_count = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        model.zero_grad()
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss_count += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_loss = total_loss / loss_count
    writer.add_scalar("Loss/train", avg_loss, epoch)


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            _, preds = torch.max(preds, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(predictions)
    writer.add_scalar("Loss/evaluate", avg_loss, epoch)
    return (accuracy_score(actual_labels, predictions),
            classification_report(actual_labels, predictions, zero_division=0),
            f1_score(actual_labels, predictions, average='weighted'),
            avg_loss)


smallest_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train(model, train_dataloader, optimizer, device)
    accuracy, report, score_f1, loss = evaluate(model, val_dataloader, device)
    writer.add_scalar("Accuracy/evaluate", accuracy, epoch)
    writer.add_scalar("Weighted F1 score/evaluate", score_f1, epoch)
    if loss <= smallest_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        smallest_loss = loss
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Average Loss: {loss:.4f}")
    print(report)

writer.flush()
torch.save(model.state_dict(), 'model.pth')
