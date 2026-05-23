from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Annotated

load_dotenv()
llm_endpoint=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text generation")
model=ChatHuggingFace(llm=llm_endpoint)


# model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',device=0)

class Validate(BaseModel):
    name:Annotated[str, Field(description="Generate a random name")]
    age:Annotated[int, Field(description="Generate a random age value", gt=18)]
    city:Annotated[str, Field(description="Generate a random city")]

parser=PydanticOutputParser(pydantic_object=Validate)

template=PromptTemplate(template="Generate the random values of a person of {place} , the information should include his/her name, age and city \n {format_instructions}",
                        input_variables=["place"],
                        partial_variables={"format_instructions":parser.get_format_instructions()})


chain=template | model | parser 

result=chain.invoke({"place":"Srilanka"})
print(result)
