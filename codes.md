![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/77bbf6e5-dd40-4a1a-8c8d-86f45b5bd820)
# Codes 
download images in a folder from saegmaker
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

