from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    model_kwargs={"device":0}
)



template1=PromptTemplate(
    template="I want you to generate a detailed linkedin post on the topic {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="I want you to write a tweet  on the topic {topic}",
    input_variables=["topic"]
)

chain=RunnableParallel({
    'tweet': template1 | model | parser,
    'linkedin_post': template2 | model | parser,
})

result=chain.invoke({"topic":"AI"})
print(result['tweet'])