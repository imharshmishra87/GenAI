from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model=GoogleGenerativeAI(model="gemini-3.1-flash-lite")

messages=[SystemMessage(content="You are a smart ai assistant"),
          HumanMessage(content="Explain Langchain")]

result=model.invoke(messages)
messages.append(AIMessage(result))

print(messages)
