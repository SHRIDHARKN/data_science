```python
import pandas as pd
import gc
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm
import os
import time
import warnings
from datasets import load_dataset
from IPython.display import Markdown
from tqdm import tqdm
import torch.optim as optim
tqdm.pandas()
gc.collect()
torch.cuda.empty_cache()
```
# Model and tokenizer context length
```python
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model_context_length = 1024
model_context_length = tokenizer.model_max_length if model_context_length is None else model_context_length
```
# Get max tokens required for answer
```python
def get_num_tokens(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

df["answer_token_len"] = df.progress_apply(lambda x:get_num_tokens(text=x["answer"], tokenizer=tokenizer),axis=1)
answer_context_length = df["answer_token_len"].max()
passage_context_length = int(0.75*(model_context_length - answer_context_length))
print(f"model_context_length:{model_context_length}")
print(f"answer_context_length:{answer_context_length}")
print(f"passage_context_length:{passage_context_length}")
print(f"passage_context_length+answer_context_length:{passage_context_length+answer_context_length}")
```
# QA template
```python
def make_qa_prompt(question,passage,answer,passage_context_length):

    passage_len = len(passage)
    text = passage if passage_len<passage_context_length else passage[:passage_context_length]
    text+="."
    text_ls = text.split(" ")[:-1]
    text = " ".join(x for x in text_ls)
    text+="."
    
    prompt = (
        "<|user|>\n"
        f"Text: {text}\n\n"
        f"Question: {question}\n"
        "<|assistant|>\n"
        f"{{\"Answer\": \"{answer}\"}}"
    )
    return prompt
```
# Model load
```python
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload = False
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=bnb_config
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

rank = 4
lora_alpha = 4

lora_config = LoraConfig(
    r=rank,
    lora_alpha=lora_alpha,
    target_modules=["qkv_proj"],  # You can inspect your model's modules for better targeting
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

model.config.use_cache = False     # ✅ Important for training
model.config.pretraining_tp = 1    # ✅ Good to set (TP = tensor parallelism, set to 1 for training)
model.config.pad_token_id = model.config.eos_token_id  # ✅ Safe for autoregressive models
```
# Train the model
```
from torch.amp import GradScaler, autocast

scaler = GradScaler("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()

num_epochs = 1

for epoch in range(num_epochs):
    num_steps = 0
    train_loss = 0
    loop = tqdm(dataloader,leave=True)
    for batch in loop:
        prompt, true_labels = batch
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", 
                   truncation=True, max_length=model_context_length).to(model.device)
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        labels = input_ids.clone().to(model.device)

        for i in range(len(prompt)):
            answer_start = prompt[i].find('{"Answer":')
            prompt_part = tokenizer(prompt[i][:answer_start], add_special_tokens=False)
            prompt_token_length = len(prompt_part["input_ids"])
            labels[i, :prompt_token_length] = -100

        optimizer.zero_grad()

        with autocast(device_type="cuda"):  # mixed precision
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        num_steps+=1
        train_loss+=loss.item()
        
    avg_train_loss = train_loss/num_steps
    print(f"Epoch {epoch+1}: Loss = {loss.item()} : Avg Train Loss : {avg_train_loss}")
```
# Method 2
```
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    epoch_loss = 0
    num_steps = 0
    loop = tqdm(dataloader,leave=True)
    for batch in loop:
        prompt, true_labels = batch
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", 
                   truncation=True, max_length=model_context_length).to(model.device)
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        labels = input_ids.clone().to(model.device)

        for i in range(len(prompt)):
            answer_start = prompt[i].find('{"Answer":')
            prompt_part = tokenizer(prompt[i][:answer_start], add_special_tokens=False)
            prompt_token_length = len(prompt_part["input_ids"])
            labels[i, :prompt_token_length] = -100

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        num_steps+=1

    avg_epoch_loss = epoch_loss/num_steps
    
    print(f"Epoch {epoch+1}: Loss = {avg_epoch_loss}")
```
