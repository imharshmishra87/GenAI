from langchain_community.tools import tool

@tool
def multiply(a:int, b:int)->int:
    "Multiply the given two numbers"
    return a*b


result=multiply.invoke({"a":5,"b":6})
print(result)
print(multiply.args)
print(multiply.description)
print(multiply.name)
