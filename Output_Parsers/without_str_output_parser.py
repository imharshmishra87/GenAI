from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm_endpoint=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text generation")
model=ChatHuggingFace(llm=llm_endpoint)

template1=PromptTemplate(template="I want you to get me indepth analysis on {topic}", 
input_variables=["topic"])

prompt1=template1.invoke({"topic":"Black Hole"})

result1=model.invoke(prompt1)


template2=PromptTemplate(template="I want you to summarize the given text {text} and summarize it into simple lines simple and clear", input_variables=['text'])

prompt2=template2.invoke({'text':result1.content})

result2=model.invoke(prompt2)

print(result2.content)
