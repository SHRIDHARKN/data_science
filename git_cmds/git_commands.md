# git config
`git config --global user.email "you@example.com"` <br>
`git config --global user.name "Your Name"`
# git commands
**list git branches**
```
git branches
```
**create a new branch**
```
git checkout -b <BRANCH_NAME>
```
**change the name of the branch**
```
git branch -m <old-name> <new-name>
```


**switch branch**
```
git checkout main
```
**delete branch locally**
```
git branch -d <BRANCH_NAME>
```
**get commit logs/ ids**
```
git log
```
**revert changes**
```
git revert <COMMIT_ID>
```
**abort merge**
```python
git merge --abort
```
**E325: ATTENTION Found a swap file by the name ".git/.COMMIT_EDITMSG.swp"** 
```
rm .git/.COMMIT_EDITMSG.swp
```
