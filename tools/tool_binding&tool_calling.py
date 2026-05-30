from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser=StrOutputParser()


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

def extract_tool_message(data):
    return data[1].tool_calls[0]

def extract(data):
    return data[0]

messages=[]
def add_message(data):
    messages.append(data)
    return messages

def usermessage(data):
    messages.append(HumanMessage(data))
    return messages

template=PromptTemplate(
    template="You are a helpfull assistant whose goal is to resolve the issue and queries -> {query} of the user",
    input_variables=["query"]
)

template2=PromptTemplate(
    template=" I want you to combine the output of both the model_message-> {Model_message} and Tool_message_extraction->{Tool_message_extraction}",

    input_variables=["Model_message","Tool_message_extraction"]
)

model_with_tools=model.bind_tools([add,multiply,divide,subtract])


query=HumanMessage("What is the sum of 5 and 6")

messages=[query]


result=model_with_tools.invoke(messages)
messages.append(result)

tool_result=add.invoke(result.tool_calls[0])
messages.append(tool_result)


final_result=model_with_tools.invoke(messages)
print(final_result)


"""query---> LLM---> Extract tool message--> Tool ---> o/p ---> LLM--> final o/p"""