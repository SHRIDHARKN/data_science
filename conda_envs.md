# conda commands
**show environment names**<br>
```
conda info --envs
```
**create a new environment**<br>
```
conda create -n <environment_name>
```
**activate environment**<br>
```
conda activate <environment_name>
```
**remove environment**<br>
```
conda env remove -n <environment_name>
```
**install specific python version**
```
conda install python=3.11
```
**check python version**
```
python --version
```
**create a env in D drive**
```
conda create --prefix D:\data_science_projects\llm\llm_env python=3.10
```
**activate env in D drive**
```
 conda activate D:\data_science_projects\llm\llm_env
```
**remove env from D drive**
```
conda env remove --prefix D:\data_science_projects\llm\llm_env
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/79a0075c-6a3f-461f-803b-23a6f1f97aa4)

**check if gpu is detected**
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
