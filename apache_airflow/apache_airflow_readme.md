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
## admin role will get created
## run the following to get started 😃
```python
docker compose up airflow
```
## localhost:8080 will have airflow running. Enter `airflow` as username and password 🏆
## stop airflow/ docker compose
```python
docker compose down
```
