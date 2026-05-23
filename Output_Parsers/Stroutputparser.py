from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',model_kwargs={"device":0})

template1=PromptTemplate(template="I want you to get me indepth analysis on {topic}", 
input_variables=["topic"])

template2=PromptTemplate(template="I want you to summarize the given text {text} and summarize it into simple lines simple and clear", input_variables=['text'])

parser=StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":'Black Hole'})

print(result)
