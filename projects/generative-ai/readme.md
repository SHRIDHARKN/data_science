# ___________________________________________________________________
## Get started with Ollama 
![image](https://github.com/user-attachments/assets/f9098cd3-e321-40f9-bf3b-005f50b1b463)

Run the following commands
- ```pip install ollama```
- ```sudo snap install ollama```
### Run llama 3.2
```ollama run llama3.2```
### List ollama pulled models
```ollama list```

# ____________________________________________________________________

## installing bitsandbytes
- run `nvcc --version` to get cuda version installed. Example 11.8 then enter 118 in following command
- `export BNB_CUDA_VERSION=118`
- run this `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
`
