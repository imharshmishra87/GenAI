from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',device=0)
result=model.invoke('What is the capital of MP')

print(result.text)