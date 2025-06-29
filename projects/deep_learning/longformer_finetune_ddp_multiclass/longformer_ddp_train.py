%%writefile ddp.py
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
import pickle

class ParquetDataset(Dataset):
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.parquet_files = [f for f in os.listdir(root_dir) if f.endswith('.parquet')]
        self.all_data = []
        self._load_metadata()
        self.num_files = len(self.parquet_files)
        print(f"Found {self.num_files} parquet files in {root_dir}")

    def _load_metadata(self):
        
        for file_idx, filename in enumerate(self.parquet_files):
            filepath = os.path.join(self.root_dir, filename)
            df = pd.read_parquet(filepath)
            for row_idx in range(len(df)):
                self.all_data.append((file_idx, row_idx))
                
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        file_idx, row_idx = self.all_data[idx]
        filename = self.parquet_files[file_idx]
        filepath = os.path.join(self.root_dir, filename)
        df = pd.read_parquet(filepath)
        sample = df.iloc[row_idx].to_dict()  # Get the row as a dictionary
        return sample

# === Hyperparameters ===
BATCH_SIZE = 8 # This will be the batch size PER GPU
EPOCHS = 3
MAX_LEN = 1024 # Standard max length for BERT-like models
MODEL_NAME = "allenai/longformer-base-4096" 
LR = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 8
PROJECT_DIR = "/kaggle/working"

# === Paths and Directories (global for easy access in config) ===
CACHE_DIR = f"{PROJECT_DIR}/pretrained_models"
TRAIN_DATA_PATH = "/kaggle/input/news-cat-train-parquet/"

MODEL_TAG = MODEL_NAME.split("/")[-1]
BASE_LOG_DIR = f"{PROJECT_DIR}/model/{MODEL_TAG}/runs/{MODEL_TAG}_classification_ddp"
BASE_CHECKPOINT_DIR = f"{PROJECT_DIR}/model/{MODEL_TAG}/checkpoints/{MODEL_TAG}_ddp"

# === Load Label Map ===
LABEL_NAME_MAP_DICT = {'automobile': 0,
 'entertainment': 1,
 'politics': 2,
 'science': 3,
 'sports': 4,
 'technology': 5,
 'world': 6}

NUM_LABELS = len(LABEL_NAME_MAP_DICT)


# === Custom Collate Function for Classification ===
# Define this at the top level so it's easily picklable and discoverable by spawned processes
def collate_fn_for_classification(batch, tokenizer_obj, max_length_obj):
    texts = [item['text'] for item in batch]
    labels_str = [item['label'] for item in batch]
    
    # Convert string labels to numerical IDs using the globally loaded map
    numerical_labels = [LABEL_NAME_MAP_DICT[label_str] for label_str in labels_str]

    tokenized_inputs = tokenizer_obj(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_length_obj,
        return_tensors="pt"
    )
    labels_tensor = torch.tensor(numerical_labels, dtype=torch.long)

    return {
        'input_ids': tokenized_inputs.input_ids,
        'attention_mask': tokenized_inputs.attention_mask,
        'labels': labels_tensor
    }

# === Setup function for each process ===
def ddp_setup(rank, world_size, master_port='12355'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# === Training Loop for each DDP process ===
def train_worker(rank, world_size, config):
    ddp_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"[{rank}/{world_size}] Process initialized on {device}")

    # --- Load Tokenizer and Model ---
    print(f"[{rank}/{world_size}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], cache_dir=config['cache_dir'])
    
    print(f"[{rank}/{world_size}] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_labels'],
        cache_dir=config['cache_dir']
    )
    
    model.to(device)
    model = DDP(model, device_ids=[rank]) # Wrap model with DDP
    print(f"[{rank}/{world_size}] Success loading tokenizer and model.")

    # --- Dataset and DataLoader ---
    dataset = ParquetDataset(config['train_data_path'])
    
    # Use partial function to pass tokenizer and max_length to collate_fn
    collate_fn_partial = lambda batch: collate_fn_for_classification(batch, tokenizer, config['max_len'])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, collate_fn=collate_fn_partial)

    # --- Optimizer and Learning Rate Scheduler ---
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Calculate total training steps for the scheduler
    total_effective_batches_per_epoch = len(dataset) / (config['batch_size'] * world_size * config['gradient_accumulation_steps'])
    num_training_steps = int(total_effective_batches_per_epoch * config['epochs'])
    if num_training_steps == 0:
        num_training_steps = 1 # Ensure at least one step if dataset is tiny

    num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # --- TensorBoard SummaryWriter (only on rank 0) ---
    writer = None
    if rank == 0:
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=config['log_dir'])
        print(f"Logs will be written to {config['log_dir']} by Rank 0")
        print(f"Checkpoints will be saved to {config['checkpoint_dir']} by Rank 0")

    # --- Training Loop ---
    model.train()
    global_step = 0
    print(f"[{rank}/{world_size}] Starting training loop...")

    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch) # Important for DistributedSampler to shuffle correctly each epoch
        
        total_loss_epoch = 0
        batch_count_epoch = 0
        optimizer.zero_grad() # Initialize gradients for accumulation

        # tqdm for rank 0 only
        pbar = tqdm(dataloader, desc=f"Rank {rank} Epoch {epoch+1}", 
                    disable=(rank != 0), leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / config['gradient_accumulation_steps'] # Normalize loss for accumulation
            loss.backward()

            # Perform optimizer step every `gradient_accumulation_steps` batches or at the end of epoch
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient clipping (optional but recommended)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients
                global_step += 1
                
                if rank == 0:
                    writer.add_scalar('Loss/train_step', loss.item() * config['gradient_accumulation_steps'], global_step)
                    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            total_loss_epoch += loss.item() * config['gradient_accumulation_steps'] # Accumulate actual loss
            batch_count_epoch += 1
            
            # Update tqdm postfix for rank 0
            if rank == 0:
                pbar.set_postfix({"loss": total_loss_epoch / max(1, batch_count_epoch)})

        # Synchronize all processes before potentially saving/logging epoch metrics
        dist.barrier() 

        # Calculate average epoch loss (global average)
        avg_epoch_loss_local = total_loss_epoch / max(1, batch_count_epoch) # Local avg
        loss_tensor = torch.tensor([avg_epoch_loss_local], device=device)
        gathered_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_epoch_loss_global = sum(g.item() for g in gathered_losses) / world_size

        if rank == 0:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss_global, epoch)
            print(f"\nEpoch {epoch+1}/{config['epochs']} finished. Average Global Loss: {avg_epoch_loss_global:.4f}")

            # Save model checkpoint (only rank 0 saves to prevent file conflicts)
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"model_checkpoint_epoch_{epoch+1}.pth")
            # Save the base model's state_dict, not the DDP wrapper
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss_global,
                'global_step': global_step,
                # Store config params for reproducibility
                'config': config 
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    if rank == 0:
        writer.close()
    
    cleanup() # Clean up DDP process group

# === Entry point ===
def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs found. DDP requires GPUs. Exiting.")
        sys.exit(1)
    else:
        print(f"Found {world_size} GPUs. Launching DDP training.")

    # Centralized configuration dictionary to pass to each worker
    config = {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'cache_dir': CACHE_DIR,
        'train_data_path': TRAIN_DATA_PATH,
        'num_labels': NUM_LABELS, # Pass num_labels directly
        'log_dir': BASE_LOG_DIR,
        'checkpoint_dir': BASE_CHECKPOINT_DIR
    }

    mp.spawn(train_worker,
             args=(world_size, config), # Arguments passed to train_worker
             nprocs=world_size,        # Number of processes to launch (one per GPU)
             join=True)                # Wait for all processes to complete

if __name__ == "__main__":
    main()
