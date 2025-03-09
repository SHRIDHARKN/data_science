# Install kubeflow
### Prerequisites
- KIND (Kubernetes in Docker installed)
## Check the clusters available
```
kind get clusters
```
## If there are no clusters , then create a new cluster for kubeflow.
```
kind create cluster --name voldemort-kubeflow-cluster
```
Running `kind get clusters` list all the clusters running. Check the active cluster.
## Get name of the active cluster
```
kubectl config current-context
```
**This should display `kind-voldemort-kubeflow-cluster`.**
## List the namespaces within the cluster.
```
kubectl get namespaces
```
**There is no namespace called `kubeflow`. Lets install it.**
## Clone kubeflow repo
```
cd ~
```
```
git clone https://github.com/kubeflow/manifests.git
```
```
git checkout v1.8-branch
```
**This will take lot of time.**
```
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 20; done
```
**Rerun the following command.**
```
kubectl get namespaces
```
**You should see kubeflow as namespace.**
## Run kubeflow
```
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
