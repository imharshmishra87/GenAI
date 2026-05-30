from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def add(a:int, b:int)->int:
    "Addition of two numbers"
    return a+b

@tool
def subtract(a:int, b:int)->int:
    "Substraction of two numbers"
    return a-b

@tool
def multiply(a:int, b:int)->int:
    "Multiplication of two numbers"
    return a*b

@tool
def divide(a:int, b:int)->int:
    "Division of two numbers"
    return a/b


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model_with_tool=llm.bind_tools([add,subtract,divide,multiply])

query="Add 3 and 4"

result=model_with_tool.invoke(query)
print(result)



# class Mathtoolkit:
#     def get_tools(self):
#         return [add,multiply,subtract,divide]
    
# obj=Mathtoolkit()
# result=obj.get_tools()
# for tools in result:
#     print(f"{tools.name} => {tools.description}")