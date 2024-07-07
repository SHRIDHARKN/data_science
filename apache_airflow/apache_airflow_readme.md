# apache airflow
## create docker compose file for airflow
```python
curl https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml -o docker-compose.yaml
```
## run the following commands
```python
mkdir dags
```
```python
mkdir logs
```
```python
mkdir plugins
```
```python
docker compose up airflow-init
```


**install apache airflow**
```
curl 'https://airflow.apache.org/docs/apache-airflow/2.6.2/docker-compose.yaml' -o 'docker-compose.yaml'
```
**paste the following commands to make required directories**
```
```
**_create a .dockerignore file and mention .git in it_**<br>
_this is to ignore downloading large files and avoid running into endless docker image building process_<br>
<br>
**build the docker image**
```
```
**start airflow services**
* start the docker 
```
docker compose up
```
**run the following command in new terminal to check health status of container services**
```
docker ps
```
*_open localhost:8080 to start airflow interface_*<br>
<br>
**stop airflow services**
```
docker compose down
```
