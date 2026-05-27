from typing import TypedDict

class Check(TypedDict):
    name:str
    age:int

data : Check ={
    'name':'Harsh',
    'age':'twelve'
}
print(data)