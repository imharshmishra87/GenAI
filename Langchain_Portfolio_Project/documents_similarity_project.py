from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated



app=FastAPI()

class INPUT(BaseModel):
    input:Annotated[str,Field(description="Enter Your Query")]

@app.get('/')
def home():
    return {"message":"Model is working completely fine"}

@app.post('/chat')
def chat(input_val:INPUT):
    load_dotenv()
    model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':0})

    cricketer_sentences = [
        "Sir Don Bradman, an Australian cricketing legend, retired with a historic test batting average of 99.94.",
        "Sachin Tendulkar, often referred to as the 'Master Blaster', has scored 100 international centuries.",
        "Virat Kohli became the fastest batsman to reach 13,000 runs in One Day Internationals.",
        "Ellyse Perry is a prominent Australian all-rounder who excelled in both cricket and association football.",
        "Muthiah Muralidaran holds the world record for the highest number of wickets in test cricket, with 800 dismissals.",
        "Ben Stokes led England to a dramatic victory in the 2019 Cricket World Cup final at Lord's.",
        "MS Dhoni is widely celebrated for his tactical captaincy and finishing games under intense pressure"
    ]

    query=input_val.input

    embedding=model.embed_documents(cricketer_sentences)

    user_embedding=model.embed_query(query)

    result=cosine_similarity([user_embedding],embedding)[0]

    index, score=sorted(list(enumerate(result)), key=lambda x:x[1])[-1]

    return JSONResponse(status_code=201, content={
        "User Query":query,
        "Response":cricketer_sentences[index],
        "Confidence Score":score
    })

