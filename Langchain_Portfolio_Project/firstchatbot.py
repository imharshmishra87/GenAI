from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm_endpoint=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text generation")
model=ChatHuggingFace(llm=llm_endpoint)


messages=[SystemMessage("You are a chat assistant whose goal is to be a helpfull with the users")]

while True:
    query=input("you:")
    messages.append(HumanMessage(query))
    if query=="exit":
        break
    result=model.invoke(messages)
    messages.append(AIMessage(result.content))
    print(f"AI: {result.content}")

print(messages)

