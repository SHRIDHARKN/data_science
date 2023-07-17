# _apache airflow_
**install apache airflow**
```
curl 'https://airflow.apache.org/docs/apache-airflow/2.6.2/docker-compose.yaml' -o 'docker-compose.yaml'
```
**paste the following commands to make required directories**
```
mkdir dags
mkdir logs
mkdir plugins
```
**_create a .dockerignore file and mention .git in it_**<br>
_this is to ignore downloading large files and avoid running into endless docker image building process_<br>
<br>
**build the docker image**
```
docker compose up airflow-init
```
**start airflow services**
```
docker compose up
```
**run the following command in new terminal to check health status of container services**
```
docker ps
```
*_open localhost:8080 to open airflow interface_*<br>
<br>
**stop airflow services**
```
docker compose down
```
