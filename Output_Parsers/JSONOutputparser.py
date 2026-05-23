from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm_endpoint=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text generation")
model=ChatHuggingFace(llm=llm_endpoint)

parser=JsonOutputParser()


template=PromptTemplate(template="Give me the name, age and city of the fictional characters {format_instructions} choose a random character on your own",
input_variables=[],
partial_variables={'format_instructions':parser.get_format_instructions()})

chain = template | model | parser

result=chain.invoke({})


print(result)
