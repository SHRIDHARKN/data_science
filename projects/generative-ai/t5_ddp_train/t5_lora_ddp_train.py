%%writefile ddp.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import PEFT components
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
warnings.filterwarnings("ignore")

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


# Assume ParquetDataset is defined elsewhere and works correctly
# from your_data_module import ParquetDataset

# --- Configuration (Moved global-like variables to be accessed from train_and_evaluate directly) ---
# These variables will be passed to main_worker or accessed within it.
# We make them global-like constants here, but they are read inside the main worker function.
GLOBAL_LR = 2e-5
GLOBAL_WEIGHT_DECAY = 0.01
GLOBAL_NUM_EPOCHS = 3
GLOBAL_GRADIENT_ACCUMULATION_STEPS = 8 # This is per-process accumulation
GLOBAL_BATCH_SIZE = 8 # This is PER-GPU batch size
GLOBAL_MODEL_CONTEXT_LENGTH = 512
GLOBAL_LABEL_CONTEXT_LENGTH = 5
checkpoint_save_epochs = 1 # Added to constants
GLOBAL_PROJ_FOLDER = "/kaggle/working"
GLOBAL_TRAIN_DATA_PATH = "/kaggle/input/news-cat-train-parquet/"
GLOBAL_MODEL_NAME = "google/flan-t5-base"
GLOBAL_MODEL_TAG = GLOBAL_MODEL_NAME.split("/")[-1]
GLOBAL_CACHE_DIR = f"{GLOBAL_PROJ_FOLDER}/pretrained_models"
GLOBAL_TEST_DATA_PATH = "/kaggle/input/news-cat-test" # Added for clarity

# LoRA Configuration
GLOBAL_LORA_R = 16 # Rank of the update matrices
GLOBAL_LORA_ALPHA = 32 # LoRA scaling factor
GLOBAL_LORA_DROPOUT = 0.05 # Dropout probability for LoRA layers
# Target modules for T5 for LoRA are typically 'q', 'v' (query, value),
# but sometimes 'k' (key) or 'o' (output) are also included.
# For FLAN-T5, 'q', 'v' is a common choice.
GLOBAL_LORA_TARGET_MODULES = ["q", "v"] 


# --- DDP specific setup ---
def setup_ddp(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost' # Or the IP of your master node
    os.environ['MASTER_PORT'] = '12355' # Choose a free port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Set the GPU for this process

def cleanup_ddp():
    """Destroys the distributed environment."""
    dist.destroy_process_group()

# --- REVISED: Collate Function as a Top-Level Function ---
# It takes the tokenizer and lengths directly as arguments.
# The tokenizer will be loaded *within each DataLoader worker process* if num_workers > 0
# OR passed explicitly to this function.
def t5_collate_fn_top_level(batch, tokenizer_instance, model_max_length, label_max_length):
    """
    Custom collate function for T5 models, designed to be picklable.
    It expects a tokenizer_instance to be passed to it.
    """
    ques = "What is the catgory of the following news article?"
    input_texts = [ques + item['text'] for item in batch]
    answer_texts = [item['label'] for item in batch]

    tokenized_inputs = tokenizer_instance( # Use the passed tokenizer_instance
        input_texts,
        padding="longest",
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt"
    )

    tokenized_answers = tokenizer_instance( # Use the passed tokenizer_instance
        answer_texts,
        padding="longest",
        truncation=True,
        max_length=label_max_length,
        return_tensors="pt"
    )

    labels = tokenized_answers.input_ids
    labels[labels == tokenizer_instance.pad_token_id] = -100

    return {
        'input_ids': tokenized_inputs.input_ids,
        'attention_mask': tokenized_inputs.attention_mask,
        'labels': labels,
        'text': input_texts,
        'label': answer_texts
    }

# --- DataLoader Worker Initialization Function ---
# This function is called by each DataLoader worker process when it starts.
# It's a perfect place to load objects that are not easily picklable,
# ensuring each worker has its own instance.
worker_tokenizer = None # To store tokenizer for workers
def worker_init_fn(worker_id):
    """
    Initializes each DataLoader worker with its own tokenizer instance.
    """
    global worker_tokenizer
    # Ensure tokenizer is only loaded once per worker
    if worker_tokenizer is None:
        worker_tokenizer = AutoTokenizer.from_pretrained(GLOBAL_MODEL_NAME, cache_dir=GLOBAL_CACHE_DIR)
    # This also handles potential issues with tokenizers and multiprocessing if not explicitly passed
    # and ensures that each worker has its own tokenizer.

def train_and_evaluate(rank, world_size):
    """
    Main training and evaluation function for a single DDP process.
    """
    setup_ddp(rank, world_size)

    # Define paths relative to each process's output
    # For LoRA, we'll save the PEFT adapters, not the full model, so modify checkpoint_dir
    log_dir = f"{GLOBAL_PROJ_FOLDER}/news_category/model/{GLOBAL_MODEL_TAG}_lora/runs/{GLOBAL_MODEL_TAG}_classification_rank_{rank}"
    checkpoint_dir = f"{GLOBAL_PROJ_FOLDER}/news_category/model/{GLOBAL_MODEL_TAG}_lora/checkpoints" # LoRA checkpoints will be here
    
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier() 

    # --- Load Tokenizer (for the main process and collate_fn argument) ---
    # Each DDP process will load its own tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(GLOBAL_MODEL_NAME, cache_dir=GLOBAL_CACHE_DIR)

    # --- Load Data ---
    train_dataset = ParquetDataset(GLOBAL_TRAIN_DATA_PATH)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    from functools import partial
    collate_fn_final = partial(t5_collate_fn_top_level,
                               tokenizer_instance=tokenizer, # This is the tokenizer of the main process
                               model_max_length=GLOBAL_MODEL_CONTEXT_LENGTH,
                               label_max_length=GLOBAL_LABEL_CONTEXT_LENGTH)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn_final, # Use the partial'd collate_fn
        num_workers=4, # Keep num_workers > 0 for efficiency
        pin_memory=True,
        worker_init_fn=worker_init_fn # Crucial for worker-specific setup
    )

    # --- Load Base Model ---
    model = AutoModelForSeq2SeqLM.from_pretrained(GLOBAL_MODEL_NAME, cache_dir=GLOBAL_CACHE_DIR)
    
    # --- Configure LoRA and Wrap Model ---
    lora_config = LoraConfig(
        r=GLOBAL_LORA_R,
        lora_alpha=GLOBAL_LORA_ALPHA,
        target_modules=GLOBAL_LORA_TARGET_MODULES,
        lora_dropout=GLOBAL_LORA_DROPOUT,
        bias="none", # T5 usually doesn't have trainable bias in LoRA
        task_type=TaskType.SEQ_2_SEQ_LM # Specify task type for T5
    )
    
    # Wrap the base model with LoRA adapters
    # This `model` will now only train the LoRA weights
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters to verify LoRA is working
    if rank == 0:
        lora_model.print_trainable_parameters()

    # --- DDP with LoRA Model ---
    # Wrap the PEFT model with DDP
    ddp_model = DDP(lora_model.to(rank), device_ids=[rank])

    # --- Optimizer and Learning Rate Scheduler ---
    # Optimizer should only optimize the trainable (LoRA) parameters
    optimizer = AdamW(ddp_model.parameters(), lr=GLOBAL_LR, weight_decay=GLOBAL_WEIGHT_DECAY)
    num_training_steps = GLOBAL_NUM_EPOCHS * (len(train_dataloader) // GLOBAL_GRADIENT_ACCUMULATION_STEPS)
    num_warmup_steps = int(num_training_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- TensorBoard SummaryWriter ---
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    # --- Training Loop ---
    ddp_model.train() 

    global_step = 0
    for epoch in range(GLOBAL_NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        
        batch_idx = 0
        total_loss = 0
        optimizer.zero_grad()

        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{GLOBAL_NUM_EPOCHS} (Rank {rank})", leave=True, disable=(rank != 0))
        for batch in loop:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            outputs = ddp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / GLOBAL_GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (batch_idx + 1) % GLOBAL_GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0:
                    writer.add_scalar('Loss/train_step', loss.item() * GLOBAL_GRADIENT_ACCUMULATION_STEPS, global_step)
                    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            total_loss += loss.item() * GLOBAL_GRADIENT_ACCUMULATION_STEPS
            batch_idx += 1
            
        dist.barrier() 

        reduced_loss = torch.tensor(total_loss).to(rank)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM) 
        
        effective_batches_per_process = len(train_dataloader) // GLOBAL_GRADIENT_ACCUMULATION_STEPS
        if effective_batches_per_process == 0:
            effective_batches_per_process = 1 
        avg_epoch_loss = reduced_loss.item() / (effective_batches_per_process * world_size)

        if rank == 0:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            print(f"Epoch {epoch+1}/{GLOBAL_NUM_EPOCHS}, Average Global Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % checkpoint_save_epochs == 0 and rank == 0:
            # Save only the PEFT adapters using save_pretrained
            # This saves adapter_config.json and adapter_model.safetensors
            lora_checkpoint_path = os.path.join(checkpoint_dir, f"lora_adapters_epoch_{epoch+1}")
            ddp_model.module.save_pretrained(lora_checkpoint_path)
            # You might want to save optimizer/scheduler state separately if needed for resuming full training
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
            }, os.path.join(lora_checkpoint_path, "optimizer_scheduler_states.pt")) # Save these separately
            print(f"LoRA adapters and states saved to {lora_checkpoint_path}")
        
        dist.barrier()


    if rank == 0:
        print("\nTraining complete!")
        if writer:
            writer.close()

    # --- Evaluation Phase (Only on rank 0) ---
    if rank == 0:
        test_dataset = ParquetDataset(GLOBAL_TEST_DATA_PATH)
        inference_tokenizer = AutoTokenizer.from_pretrained(GLOBAL_MODEL_NAME, cache_dir=GLOBAL_CACHE_DIR)
        
        test_collate_fn_final = partial(t5_collate_fn_top_level,
                                        tokenizer_instance=inference_tokenizer,
                                        model_max_length=GLOBAL_MODEL_CONTEXT_LENGTH,
                                        label_max_length=GLOBAL_LABEL_CONTEXT_LENGTH)

        test_dataloader = DataLoader(test_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, collate_fn=test_collate_fn_final)

        # Load the base model first
        base_model_for_inference = AutoModelForSeq2SeqLM.from_pretrained(GLOBAL_MODEL_NAME, cache_dir=GLOBAL_CACHE_DIR)
        
        # Find the latest LoRA checkpoint directory
        lora_checkpoint_dirs = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f)) and f.startswith('lora_adapters_epoch_')]
        
        if lora_checkpoint_dirs:
            # Sort by epoch number to get the latest
            latest_lora_checkpoint_dir = max(lora_checkpoint_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))
            print(f"Loading LoRA adapters for evaluation from {latest_lora_checkpoint_dir}")
            
            # Load the PEFT model from the base model and the adapter weights
            inference_model = PeftModel.from_pretrained(base_model_for_inference, latest_lora_checkpoint_dir)
            inference_model = inference_model.merge_and_unload() # Merge LoRA weights into base model for faster inference
            inference_model.eval()
            print(f"LoRA adapters loaded and merged successfully for evaluation.")
        else:
            print("No LoRA checkpoints found for evaluation.")
            cleanup_ddp()
            return

        inference_model.to(rank)

        ques = "What is the catgory of the following news article?"

        preds = []
        acts = []

        loop = tqdm(test_dataloader, desc="Inference", leave=False, disable=False)
        for batch in loop:
            input_text = [ques + t for t in batch['text']]
            inputs = inference_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=GLOBAL_MODEL_CONTEXT_LENGTH).to(rank)

            with torch.no_grad():
                outputs = inference_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=GLOBAL_LABEL_CONTEXT_LENGTH
                )

            generated_text = inference_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend(generated_text)
            acts.extend(batch['label'])
            
        df_results = pd.DataFrame({
            'predicted_label': preds,
            'actual_label': acts
        })

        print(df_results.head())
        accuracy = accuracy_score(df_results['actual_label'], df_results['predicted_label'])
        precision, recall, f1, _ = precision_recall_fscore_support(df_results['actual_label'], df_results['predicted_label'], average='weighted', zero_division=0)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        df_overall = pd.DataFrame({
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1]
        })

        results_dir = f'{GLOBAL_PROJ_FOLDER}/news_category/model/{GLOBAL_MODEL_TAG}_lora/results'
        os.makedirs(results_dir, exist_ok=True)
        df_overall.to_csv(f'{results_dir}/overall_metrics.csv', index=False)
        df_results.to_csv(f'{results_dir}/results.csv', index=False)

    cleanup_ddp()

# --- Main execution block for DDP ---
def main():
    
    world_size = torch.cuda.device_count() # Get number of available GPUs
    if world_size == 0:
        print("No GPUs found. Running on CPU (DDP not applicable).")
        # For CPU, you'd run the single-GPU version of the code, or handle it as world_size=1
        train_and_evaluate(0, 1) # Run single process for CPU
    else:
        print(f"Found {world_size} GPUs. Spawning processes for DDP.")
        mp.spawn(train_and_evaluate, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
