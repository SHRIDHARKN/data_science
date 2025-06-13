import sys
import os
sys.path.append(os.path.abspath(".")) # Adjust the path to your project root if necessary
from utils.data_utils import ParquetDataset # Assuming you have a custom dataset class for loading parquet files

import torch
torch.cuda.empty_cache() # Clear CUDA cache to free up memory
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm # For a nice progress bar in the training loop
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

# --- Configuration ---
batch_size = 8
lr = 2e-5
weight_decay = 0.01
num_epochs = 3
gradient_accumulation_steps = 8
checkpoint_save_epochs = 1 # Save checkpoint every X epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_context_length = 512


train_data_path = "/mnt/d/data/news_category/preprocessed/train/"
model_name = "bert-base-uncased" # Changed to a BERT-like model for classification
model_tag = model_name.split("/")[-1] # Extract model name for logging
cache_dir = "/mnt/d/pretrained_models"
log_dir = f"/mnt/d/data/news_category/model/{model_tag}/runs/{model_tag}_classification" # Directory for TensorBoard logs
checkpoint_dir = f"/mnt/d/data/news_category/model/{model_tag}/checkpoints/{model_tag}" # Directory to save model checkpoints

# Ensure directories exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Map numerical labels to descriptive text labels (Adjust for your news categories)
# IMPORTANT: Replace this with your actual news categories and their corresponding numerical labels.
# load label map 
with open("/mnt/d/data/news_category/constants/label_map.pkl", "rb") as f:
    label_name_map_dict = pickle.load(f)

num_labels = len(label_name_map_dict) # Number of unique categories for BERT's classification head

# # --- Custom Collate Function for BERT Classification ---
def bert_collate_fn(batch, tokenizer, max_length):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    labels = [label_name_map_dict[label] for label in labels]  # Convert numerical labels to text labels

    # Tokenize inputs
    tokenized_inputs = tokenizer(
        texts,
        padding="longest", # Pad to the longest sequence in the batch
        truncation=True,   # Truncate sequences longer than max_length
        max_length=max_length,
        return_tensors="pt" # Return PyTorch tensors
    )

    # Convert labels to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': tokenized_inputs.input_ids,
        'attention_mask': tokenized_inputs.attention_mask,
        'labels': labels_tensor
    }

# # --- Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# For classification, we use AutoModelForSequenceClassification and specify num_labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels, # Pass the number of unique labels
    cache_dir=cache_dir
)

model.to(device)

# # --- Dataloader Initialization ---
# # BERT-based models typically use a max length of 512

train_dataset = ParquetDataset(train_data_path)
# # Pass relevant arguments to collate_fn
collate_fn_partial = lambda batch: bert_collate_fn(batch, tokenizer, model_context_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_partial)

# # --- Optimizer and Learning Rate Scheduler ---
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# # Calculate total training steps for the scheduler
# # Assuming len(train_dataset) gives total number of samples for correct num_training_steps calculation

# Approximate number of batches per epoch
num_training_steps = num_epochs * (len(train_dataset) // batch_size // gradient_accumulation_steps)
# Add a small warmup (e.g., 10% of total steps)
num_warmup_steps = int(num_training_steps * 0.1)


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# --- TensorBoard SummaryWriter ---
writer = SummaryWriter(log_dir=log_dir)

# --- Training Loop ---
model.train() # Set the model to training mode

global_step = 0
print(f"Starting training on {device}...")
for epoch in range(num_epochs):
    total_loss = 0
    batch_idx = 0
    optimizer.zero_grad() # Initialize gradients for the accumulation

    # Use tqdm for a nice progress bar
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    #loop = tqdm(train_dataloader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) # Numerical labels for classification

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels # BERT's loss is calculated internally when labels are provided
        )

        loss = outputs.loss / gradient_accumulation_steps # Normalize loss for gradient accumulation
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step() # Update learning rate
            optimizer.zero_grad() # Clear gradients
            global_step += 1

            # Log learning rate and loss to TensorBoard
            writer.add_scalar('Loss/train_step', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

        total_loss += loss.item() * gradient_accumulation_steps # Accumulate actual loss
        batch_idx += 1
        
    avg_epoch_loss = total_loss / (len(train_dataloader) / gradient_accumulation_steps) # Adjust for accumulated batches
    writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
    print(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")

    # Save model checkpoint every X epochs
    if (epoch + 1) % checkpoint_save_epochs == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        # Save both model state_dict and optimizer/scheduler states for full resume capability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'global_step': global_step,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'model_name': model_name,
            'num_labels': num_labels,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

print("\nTraining complete!")
writer.close() # Close the TensorBoard writer
