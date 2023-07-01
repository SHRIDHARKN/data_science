# _apache airflow_
**install apache airflow**
```
pip install "apache-airflow==2.6.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.2/constraints-no-providers-3.11.txt"
```
**initialise airflow db**
```
airflow db init
```
**create a airflow user**
```
airflow users create  --username admin  --firstname <FRIST_NAME>  --lastname <LAST_NAME>  --role Admin --email None
```
