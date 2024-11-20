# setup project folders
```python
import os
# create project structure
def setup_project_structure(root_dir="."):
    
    print("NOTE: Folders would be created @ current directory. If you wish to create the folders else where, then use root_dir=<YOUR PATH>")
    # Define the folder structure
    folders = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "models",
        "reports/figures",
        "reports/metrics",
        "tests",
        "configs"
    ]

    # Create the folders
    for folder in folders:
        os.makedirs(os.path.join(root_dir, folder), exist_ok=True)

    # Define essential files in the project root
    files = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "environment.yml",
        "setup.py",
        "main.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]

    for file in files:
        with open(os.path.join(root_dir, file), "w") as f:
            pass  # Creates an empty file

    # Add example content to README.md
    with open(os.path.join(root_dir, "README.md"), "w") as f:
        f.write("# Project Name\n\nThis project is a data science project.\n\n## Folder Structure\n\n")

    print(f"Folder structure created successfully in {os.path.abspath(root_dir)}!")

```
## code for data validation - pandas
```python
import pandas as pd
import numpy as np

def data_validation(reference_data: pd.DataFrame, target_data: pd.DataFrame, id_column: str):
    # Set index to the identifier column for easier comparison (assuming `id_column` uniquely identifies rows)
    reference_data = reference_data.set_index(id_column)
    target_data = target_data.set_index(id_column)
    
    # Ensure columns in both DataFrames match
    common_columns = reference_data.columns.intersection(target_data.columns)
    reference_data = reference_data[common_columns]
    target_data = target_data[common_columns]

    # Align data types in each column
    for col in common_columns:
        if reference_data[col].dtype != target_data[col].dtype:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                reference_data[col] = pd.to_numeric(reference_data[col], errors='coerce')
                target_data[col] = pd.to_numeric(target_data[col], errors='coerce')
            elif pd.api.types.is_datetime64_any_dtype(reference_data[col]):
                reference_data[col] = pd.to_datetime(reference_data[col], errors='coerce')
                target_data[col] = pd.to_datetime(target_data[col], errors='coerce')
            else:
                reference_data[col] = reference_data[col].astype(str)
                target_data[col] = target_data[col].astype(str)

    # Fill NaN values with a placeholder to ensure consistent comparison
    reference_data = reference_data.fillna("NaN_placeholder")
    target_data = target_data.fillna("NaN_placeholder")
    
    # Align target rows with reference data by index (id_column)
    target_data = target_data.reindex(reference_data.index)

    # Dictionary to store mismatches by column
    mismatches = {col: [] for col in common_columns}

    # Check for matches and mismatches
    for col in common_columns:
        mismatched_indices = reference_data.index[reference_data[col] != target_data[col]].tolist()
        mismatches[col] = mismatched_indices

    # Count total matches and mismatches
    num_matches = len(reference_data) - sum(len(indices) for indices in mismatches.values())
    num_mismatches = len(reference_data) - num_matches

    # Filter out columns with no mismatches
    mismatches = {col: indices for col, indices in mismatches.items() if indices}

    # Result summary
    result = {
        "Number of Records Matched": num_matches,
        "Number of Records Mismatched": num_mismatches,
        "Mismatches by Column": mismatches
    }
    return result

```
## BERT - classification - imbalanced class
```python
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter

# Load IMDB dataset
dataset = load_dataset("imdb")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert to torch dataset
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Compute class weights
label_counts = Counter(tokenized_datasets['train']['label'].numpy())
total_count = len(tokenized_datasets['train'])
class_weights = {label: total_count / count for label, count in label_counts.items()}

# Create WeightedRandomSampler
weights = [class_weights[label.item()] for label in tokenized_datasets['train']['label']]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Create DataLoader
train_loader = DataLoader(tokenized_datasets['train'], sampler=sampler, batch_size=8)
eval_loader = DataLoader(tokenized_datasets['test'], sampler=SequentialSampler(tokenized_datasets['test']), batch_size=8)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training function
def train(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, eval_loader, device):
    model.eval()
    preds, true_labels = [], []
    for batch in eval_loader:
        with torch.no_grad():
            inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary')
    return acc, precision, recall, f1

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 3 epochs
    train_loss = train(model, train_loader, optimizer, scheduler, device)
    acc, precision, recall, f1 = evaluate(model, eval_loader, device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}")


```
