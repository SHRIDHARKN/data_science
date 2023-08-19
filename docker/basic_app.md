# let's build a simple app that takes 2 inputs and prints the same
create a app.py file and paste the following code:<br>
```python
def myfunc(name,city):
    return f"hi : {name} from {city}"

if __name__ == "__main__":
    name = input("Please enter your name: ")
    city = input("Please enter your city: ")
    result = myfunc(name,city)
    print(result)
```
