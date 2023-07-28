```
# Use the official Python image as the base image
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run the Python script when the container launches
CMD ["python", "app.py"]
```
# build docker image <br>
```
docker build -t simple-python-app .
```
# list docker images <br>
```
docker images
```
# check the running docker containers
```
docker ps
```
