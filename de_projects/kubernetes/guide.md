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

