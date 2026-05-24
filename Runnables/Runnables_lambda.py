from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    model_kwargs={"device":0}
)


template=PromptTemplate(
    template="I want you to generate a joke on the following topic -> {topic}",
    input_variables=["topic"]
)

sequence=RunnableSequence(template, model, parser)

parallel=RunnableParallel({
    'passthrough':RunnablePassthrough(),
    'length':RunnableLambda(lambda x: len(x.split()))
})

final_chain= RunnableSequence(sequence, parallel)

result=final_chain.invoke({'topic':"AI"})
print(result)
