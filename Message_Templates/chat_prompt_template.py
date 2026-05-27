from langchain_core.prompts import ChatPromptTemplate

chat=ChatPromptTemplate([
    ("system","You are a helpful AI assistant expert of {domain}"),
    ("human","Explain me {topic} in easy and simplified language")
])

prompt=chat.invoke({
    "domain":"Physics",
    "topic":"Black Hole"
})

print(prompt)

#Use one of 'human', 'user', 'ai', 'assistant', or 'system'.