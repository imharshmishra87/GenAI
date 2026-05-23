from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model=GoogleGenerativeAI(model='gemini-3.1-flash-lite')

template1=PromptTemplate(
    template="Generate a detailed report on the {topic} use desired examples, keep it lengthy and understandable enough with simple and easy language",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="I want you to summarize the given text {text} into simple and easy language only keep essential and important points",
    input_variables=["text"]
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

print(chain.get_graph().draw_ascii())
result=chain.invoke({"topic":"attention to the model"})
print(result)