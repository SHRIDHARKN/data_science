![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/a8b1a078-8ddb-4f34-b7b7-fb7124f33f2e)

## docker
- docker image - one time template to build container
- docker container - has libraries to run the app
- container package of application with necessary dependencies
- download .exe file from offical website 
- `wsl --set-default-version 2` - use this is wsl --status shows version as 1. Docker needs wsl 2
- collection of related images - repository. Centralised location of images is registry.
- Docker - docker engine commmunicates through docker engine the apps and host OS. In VMs , app and OS kernel both needs to be setup. Hence more resources relatively. Virtualization of app only in docker.
- Docker compatibility wit windows <10

## commands
## check the docker version installed
```python
docker -v
```
## check the running docker containers
```
docker ps
```
## build docker image <br>
```
docker build -t simple-python-app .
```
## list docker images <br>
```
docker images
```
## stop docker
```python
docker stop <CONTAINER_ID>
```
# Use the official Python image as the base image
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run the Python script when the container launches
CMD ["python", "app.py"]
## see all containers, including the ones that are not running
```
docker ps -a
```
## delete docker container
```
docker rm -f <CONTAINER_ID_OR_NAME>
```
## delete docker images
```
docker rmi <IMAGE_ID_OR_NAME>
```
