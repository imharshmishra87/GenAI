from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    model_kwargs={"device":0}
)

template1=PromptTemplate(
    template="I want you to generate a detailed report on the topic -> {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="I want you to summarize the detailed report on the topic and keept it short and simple -> {text}",
    input_variables=["text"]
)

sequence_chain=RunnableSequence(template1,model,parser)
branch=RunnableBranch(
    (lambda x : len(x.split())>=500, RunnableSequence(template2, model, parser)),
    RunnablePassthrough()
)

merged= RunnableSequence(sequence_chain,branch)
result=merged.invoke({"topic":"AI"})
print(result)
