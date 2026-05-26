from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


loader=TextLoader("Document_Loaders/Data/data.txt",encoding='utf-8')

parser=StrOutputParser()

data=loader.load()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

template1=PromptTemplate(
    template="I want you to summarize the given text and give me the simplified version of it {text}",
    input_variables=["text"]
)

chain=template1 | model | parser

result=chain.invoke({'text':data[0].page_content})
print(result)