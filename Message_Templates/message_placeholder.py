from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

chat_history_template=ChatPromptTemplate([
    ('system',"You are a helpful ai refund and finance management agent"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])


chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt=chat_history_template.invoke({
    'chat_history':chat_history,
    'query':'What is my refund status ?'
})

print(prompt)