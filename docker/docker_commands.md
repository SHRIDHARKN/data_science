![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/9a12d731-9225-45d9-9ec3-73746c3549d2)

## docker
- container package of application with necessary dependencies
- ![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/862c09a9-f58c-4d89-beb7-5a59cd031f9f)
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
```
## build docker image <br>
```
docker build -t simple-python-app .
```
## list docker images <br>
```
docker images
```
## build docker file <br>
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
