<img src="https://github.com/user-attachments/assets/2503c981-521f-4a94-8f49-f4e559908015" alt="image" width="600" height="400">
# KUBERNETES
```
kubectl cluster-info
```
#### __________________________________________________________________________________________
## cluster
#### create a cluster
```
kind create cluster --name voldemort-cluster
```
#### name of the current cluster
```
kubectl config current-context
```
#### __________________________________________________________________________________________

## namespace
#### create a namespace
```
kubectl create ns voldemort-namespace
```
#### delete a namespace
```
kubectl delete ns voldemort-namespace
```
#### __________________________________________________________________________________________
## pods
#### create a pod
- create a pod.yaml file
```
apiVersion: v1
kind: Pod
metadata:
  name: voldemort-redis-pod
  namespace: voldemort-namespace
spec:
  containers:
  - name: voldemort-redis-container
    image: redis:latest
```
#### spin pod
```
kubectl apply -f pod.yaml --namespace=voldemort-namespace
```
#### list the pods running
```
kubectl get pods -n voldemort-namespace
```
#### delete pods
```
kubectl delete pod voldemort-redis-pod --namespace=voldemort-namespace
```
#### alternate way to spin pod
```
kubectl run voldemort-redis-pod --image=redis:latest --namespace=voldemort-namespace restart=Never
```
#### __________________________________________________________________________________________
