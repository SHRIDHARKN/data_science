![image](https://github.com/user-attachments/assets/d9be7c0a-8c70-4e49-9cf2-1a63b912f128)

## installation
### start wsl
### run the following commands
```
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
```
```
chmod +x ./kind
```
```
sudo mv ./kind /usr/local/bin/kind
```
### verify installation
```
kind --version
```
### create cluster
```
kind create cluster --name <project-name>
```
### list clusters
```
kind get clusters
```
### delete cluster
```
kind delete cluster --name <project-name>
```
### check if cluster is running
```
kubectl cluster-info --context kind-<cluster-name>
```
### create a namespace
```
kubectl create namespace <project-name>
```
### list namespaces
```
kubectl get namespaces
```

### create a pod
#### example - spark image base
```
kubectl run preprocess --image=bitnami/spark --namespace=ml-proj-1 --restart=Never
```
#### tensorflow image
```
kubectl run training --image=tensorflow/tensorflow:latest --namespace=ml-proj-1 --restart=Never
```
#### note: multiple pods can be created
### list pods
```
kubectl get pods
```
### list pods in namespace
```
kubectl get pods --namespace=ml-proj-1
```
### {color:red}kubectl top nodes gives error: Metrics API not available{color}
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```
```
kubectl get pods -n kube-system
```
```
kubectl edit deployment metrics-server -n kube-system
```
add this below args section
```
- --kubelet-insecure-tls
- --kubelet-preferred-address-types=InternalIP,Hostname,ExternalIP
```
```
kubectl rollout restart deployment metrics-server -n kube-system
```
```
kubectl top nodes
```
















