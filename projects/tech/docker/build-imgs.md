# Steps
- create a dockerfile
- create requirements file
- build docker image
```
docker build -t voldemort-spark-gpu .
```
- docker login and enter the otp in browser
- push the docker image to hub
```
docker tag voldemort-spark-gpu <username>/voldemort-spark-gpu
```
docker push <username>/voldemort-spark-gpu
