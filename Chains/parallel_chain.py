from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=GoogleGenerativeAI(model='gemini-3.1-flash-lite')

llm_endpoint=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text generation")
model2=ChatHuggingFace(llm=llm_endpoint)


template1=PromptTemplate(
    template="I want you to generate detailed notes on the following text -> {text}",
    input_variables=["text"]
)

template2=PromptTemplate(
    template="I want you to generate a quiz by using the following text try to cover the complete data of the text and convert them into a high quality quiz -> {text}",
    input_variables=["text"]
)

template3=PromptTemplate(
    template="I want you to combine both the notes -> {notes} and quiz -> {quiz} into a single file",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel=RunnableParallel(
    {
        "notes": template1 | model1 | parser,
        "quiz":template2 | model2 | parser
    }
)

merge= template3 | model1 | parser

chain = parallel | merge 

text="""Built an end to end Machine Learning model using Fast API and Streamlit.

Welcome to episode one let HR and companies judge you by your work.

This time I am letting people judge my skills set based on my recent completion of a Machine Learning model. 

This Machine Learning model is trained over a dummy diabetes dataset from kaggle.

The main objective was to verify what and practically implement what I have studied during my FastAPI course.

It was a diabetes dataset consisting of columns like age, pedigree diabetes function, insulin, bmi and many others columns that reflects how diabetes can affect a person.

To build this ML model I have done two main operations: Standard Scaler and then passed the scaled dataset in the logistic Regression algorithm.

I built a pipeline which automatically scales the data and passes it into the Logistic Regression algorithm without doing any manual work.

Once the ML model is trained, Imported using pickle and then implemented the concept of FastAPI.

The data validation part was handled by the Pydantic library, such that no irregular values can be passed into our database.

In order to do that I implemented Field Validators, Model Validators which ensures that our model accepts preprocessed and valid data not raw or null data.

Pydantic ensured that age value should be greater than 0 and less than 120

Added description with the help of field and Annotated modules.

Defined valid data types for the input data (eg: Age must be int)

Used literally to provide options for Gender.

Finally after data validation the next step was to ensure a workable environment and create API endpoints.

I created 3 API endpoints: home, health and prediction

Home -> If someone hits the home endpoint they will be directed to the home page which usually consists of a message.

Health-> This end point specifies the health of the api by sending status 200 ok messages and version of the model.

Prediction -> This is the end point where the prediction of the model takes place, it takes the input data from the user that validates the data and sends the data to the model and results the output as 0 or 1.

To make the API production grade, there were several improvements done such as categorizing the output with low , medium and high risk, displaying confidence scores of other models and improving the overall visual appearance of the API.

Ps: The goal was to successfully integrate the ML model with the api backend so the model isn't much trained but the overall objective of building the projects were accomplished.

Lastly I deployed the whole project on docker hub so now anyone can pull it from docker and can access and runk the model image on their own computer system.

Please go through the docker and GitHub link of the project attached below.
"""

result=chain.invoke({"text":text})
print(chain.get_graph().draw_ascii())

print(result)