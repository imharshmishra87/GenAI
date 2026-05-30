from langchain_community.tools import StructuredTool
from pydantic import BaseModel,Field

class Multiplydata(BaseModel):
    a: int=Field(description="Enter the value of a",required=True)
    b: int=Field(description="Enter the value of b",required=True)


def multiply(a,b):
    return a*b

multiply_tool=StructuredTool(
    func=multiply,
    name="Multiply",
    args_schema=Multiplydata,
    description="A mutiplication function that takes two inputs and produce a single output"
)

print(multiply_tool.invoke({"a":5,"b":6}))
print(multiply_tool.args)
print(multiply_tool.description)