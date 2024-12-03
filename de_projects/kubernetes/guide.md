![image](https://github.com/user-attachments/assets/d9be7c0a-8c70-4e49-9cf2-1a63b912f128)

## installation
### open powershell in admin mode 
head to [chocolatey-install](https://chocolatey.org/install) OR paste the following command <br>
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
### run the following command in powershell
```
choco install kind
```
😌🏆

## getting started
### open powershell and run
```
kind --version
```
If you see the version, then kind has been installed correctly.
### run the command below to find the path for kind - needed for export path in wsl
```
where.exe kind
```
### open command prompt and start WSL
### run the following 
```
vim ~/.bashrc
```
press I, go the end and paste
```
export PATH=$PATH:/mnt/c/ProgramData/chocolatey/bin
```
press esc and then wq.
```
source ~/.bashrc
```
