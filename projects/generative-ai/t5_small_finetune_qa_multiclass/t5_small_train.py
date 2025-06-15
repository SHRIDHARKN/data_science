# load data 
import sys
import os
sys.path.append(os.path.abspath("."))
from utils.data_utils import ParquetDataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import get_cosine_schedule_with_warmup
import os
from torch.optim import AdamW # AdamW is often preferred for Transformers
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

lr = 2e-5
weight_decay = 0.01
num_epochs = 3
gradient_accumulation_steps = 8
checkpoint_save_epochs = 1
batch_size = 4
model_context_length = 512  # Maximum length for model inputs
label_context_length = 5  # Maximum length for labels

train_data_path = "/mnt/d/data/news_category/preprocessed/train/"
model_name = "google/flan-t5-small"
model_tag = model_name.split("/")[-1] # Extract model tag for saving
cache_dir = "/mnt/d/pretrained_models"
log_dir = f"/mnt/d/data/news_category/model/{model_tag}/runs/{model_tag}_classification" # Directory for TensorBoard logs
checkpoint_dir = f"/mnt/d/data/news_category/model/{model_tag}/checkpoints/{model_tag}" # Directory to save model checkpoints
# Ensure directories exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# load data
train_data_path = "/mnt/d/data/news_category/preprocessed/train/"
train_dataset = ParquetDataset(train_data_path)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# load model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

def t5_collate_fn(batch, tokenizer, model_max_length, label_max_length):
    
    ques = "What is the catgory of the following news article?"
    input_texts = [ques+item['text'] for item in batch]
    answer_texts = [item['label'] for item in batch]

    # Tokenize inputs
    tokenized_inputs = tokenizer(
        input_texts,
        padding="longest",
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt"
    )

    # Tokenize targets (answers)
    tokenized_answers = tokenizer(
        answer_texts,
        padding="longest",
        truncation=True,
        max_length=label_max_length,
        return_tensors="pt"
    )

    # Prepare labels for T5 (shifted right, pad_token_id replaced with -100)
    labels = tokenized_answers.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        'input_ids': tokenized_inputs.input_ids,
        'attention_mask': tokenized_inputs.attention_mask,
        'labels': labels,
        'text': input_texts,
        'label': answer_texts
    }
    
collate_fn_partial = lambda batch: t5_collate_fn(batch, tokenizer, model_context_length, label_context_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_partial)

# --- Optimizer and Learning Rate Scheduler ---
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Calculate total training steps for the scheduler
# Approximate number of batches per epoch
num_training_steps = num_epochs * (len(train_dataset) // batch_size // gradient_accumulation_steps)
# Add a small warmup (e.g., 10% of total steps)
num_warmup_steps = int(num_training_steps * 0.1)


# Cosine scheduler with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- TensorBoard SummaryWriter ---
writer = SummaryWriter(log_dir=log_dir)

# --- Training Loop ---
model.train() # Set the model to training mode

global_step = 0
for epoch in range(num_epochs):
    
    batch_idx = 0
    total_loss = 0
    optimizer.zero_grad() # Initialize gradients for the accumulation

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for batch in loop:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
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
        
    # Log average loss for the epoch
    avg_epoch_loss = total_loss / (len(train_dataloader) / gradient_accumulation_steps) # Adjust for accumulated batches
    writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

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
            'model_name': model_name,
            'lr': lr,
            'weight_decay': weight_decay,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'model_context_length': model_context_length,
            'label_context_length': label_context_length,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

print("\nTraining complete!")
writer.close() # Close the TensorBoard writer
