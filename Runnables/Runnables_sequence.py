from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()



model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    model_kwargs={"device":0}
)

template1=PromptTemplate(
    template="I want you to generate a joke on the following topic -> {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Give me the context about this joke -> {response}",
    input_variables=["response"]
)

sequence=RunnableSequence(template1, model, parser,template2,model,parser)

result=sequence.invoke({"topic":"unemployment"})
print(result)


