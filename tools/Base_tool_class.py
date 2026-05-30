from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import BaseModel,Field
from typing import Type

class Multiplydata(BaseModel):
    a: int=Field(description="Enter the value of a",required=True)
    b: int=Field(description="Enter the value of b",required=True)


class Multiply(BaseTool):
    name: str="multiply"
    description:str="Multiply the given two numbers a, b and results o/p"

    args_schema:Type[BaseModel]=Multiplydata

    def _run(self,a:int,b:int)->int:
        return a*b
    
tool=Multiply()

result=tool.invoke({'a':5,'b':6})
print(result)