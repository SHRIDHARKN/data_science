![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/77bbf6e5-dd40-4a1a-8c8d-86f45b5bd820)
# Codes 
### Make changes to repo
<span style="color:red;">your text here > </span>
- git checkout main
- create branch from jira
- git checkout <branch-name>
- make changes and commit
### gpu memory usage
```python

import torch

def get_gpu_memory():

if torch.cuda.is_available():
        print('Cuda available')
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated(0)
        free_memory = torch.cuda.memory_reserved(0)
        max_memory = torch.cuda.max_memory_allocated(0)
        print('Total GPU memory:', round(total_memory / (1024 * 1024 * 1024), 2), 'GB')
        print('Used GPU memory:', round(used_memory / (1024 * 1024 * 1024), 2), 'GB')
        print('Free GPU memory:', round(free_memory / (1024 * 1024 * 1024), 2), 'GB')
        print('Max allocated GPU memory:', round(max_memory / (1024 * 1024 * 1024), 2), 'GB')
    else:
        print('No GPU available')

get_gpu_memory()
```
### download images in a folder from saegmaker
```python
tar -czvf <folder_name>.tar.gz <path to folder>
```
### read images 
```python
from PIL import Image
image = Image.open("images/my_image.png")
```
## LLM / Generatiev AI
### load finetuned llm and its tokenizer
```python
## folder where finetuned llm is stored, followed by checkpoint

from transformers import AutoModelForCausalLM,AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tinyllama-sqllm-v1/checkpoint-500")
tokenizer = AutoTokenizer.from_pretrained("tinyllama-sqllm-v1/checkpoint-500")
```
## generate requirements file from py libraries info 
Pass the string with py librabries info to the function along with name of the requirements file
```python
def write_libraries_to_file(libraries_string, filename):
  with open(filename, "w") as file:
    file.write(libraries_string)
```


