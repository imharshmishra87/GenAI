from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm_endpoint=HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2',task='text-generation')
model=ChatHuggingFace(llm=llm_endpoint)

result=model.invoke('What is the capital of Gujrat')

print(result.content)