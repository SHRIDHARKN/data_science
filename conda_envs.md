![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/33871c2e-d40c-4263-bbe7-5479bc8eec07)
# conda commands
**show environment names**<br>
```python
conda info --envs
```
**create a new environment**<br>
```python
conda create -n <environment_name>
```
**activate environment**<br>
```python
conda activate <environment_name>
```
**remove environment**<br>
```python
conda env remove -n <environment_name>
```
**install specific python version**
```python
conda install python=3.11
```
**check python version**
```python
python --version
```
**remove unused packages**
```python
conda clean --yes --all
```
**create kernel in jupyter notebook**
```python
conda activate env
```
```python
pip install ipykernel
```
```python
python -m ipykernel install --user --name <environemnt name> --display-name "<display name>"
```

**create a env in D drive**
```python
conda create --prefix D:\data_science_projects\llm\llm_env python=3.10
```
**activate env in D drive**
```python
 conda activate D:\data_science_projects\llm\llm_env
```
**remove env from D drive**
```python
conda env remove --prefix D:\data_science_projects\llm\llm_env
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/79a0075c-6a3f-461f-803b-23a6f1f97aa4)<br>
### cuda setup
- Install cuda 11.4
- Set the path in system variables
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\extras\CUPTI\lib64
```
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include
```
- Extract cudnn-windows-x86_64-8.9.6.50_cuda11-archive
- create a folder C:\cudnn
- paste the contents – bin/include/lib
- set the path in system variables
```
C:\cudnn\bin
```
```
C:\cudnn\include
```
```
C:\cudnn\lib\x64
```


### cuda/gpu setup in a conda environment
```python
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
```python
python -c "import torch; print(torch.cuda.device_count())"
```
```python
conda create --prefix D:\data_science_projects\llm\llm_env python=3.10
```
```python
conda activate D:\data_science_projects\llm\llm_env
```
```python
conda install -c conda-forge cudatoolkit=11.4 cudnn=8.2
```

**check if gpu is detected**
```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
