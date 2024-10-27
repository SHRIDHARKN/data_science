# git config
`git config --global user.email "<your email id>"` <br>
`git config --global user.name "<your name/ user name>"`
# git commands
##### Stage the changes
```git add .```
##### Commit changes
```git commit -m "<message>" ```

#### list git branches
```
git branches
```
#### create a new branch
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
**merge branch with main**
- checkout to main branch and enter the following commands
```
git merge <branch-name>
```
**get commit logs/ ids**
```
git log --oneline
```
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
