# let's build a simple app that takes 2 inputs and prints the same
---
### create a app.py file and paste the following code:<br>
```python
def myfunc(name,city):
    return f"hi : {name} from {city}"

if __name__ == "__main__":
    name = input("Please enter your name: ")
    city = input("Please enter your city: ")
    result = myfunc(name,city)
    print(result)
```
### build docker image (paste the code below):<br>
```python
docker build -t simple-python-app .
```
### run docker image (paste the code below):<br>
```python
docker run -it simple-python-app
```
### prompted to enter the inputs:<br>
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/9f4f7a01-e74a-4644-9abf-7f55607cf1b9)
