from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import langchain


docs = [
    # --- 5 Genuine Points on How to Stay Healthy ---
    Document(
        page_content="Maintain a balanced diet rich in fruits, vegetables, and lean proteins to support your immune system.",
        metadata={"context": "human_health", "type": "genuine"}
    ),
    Document(
        page_content="Exercise regularly for at least 30 minutes a day to improve cardiovascular health and strengthen muscles.",
        metadata={"context": "human_health", "type": "genuine"}
    ),
    Document(
        page_content="Ensure you get 7-9 hours of quality sleep every night for proper cognitive function and physical recovery.",
        metadata={"context": "human_health", "type": "genuine"}
    ),
    Document(
        page_content="Stay hydrated by drinking plenty of water throughout the day to keep your biological systems functioning optimally.",
        metadata={"context": "human_health", "type": "genuine"}
    ),
    Document(
        page_content="Manage daily stress through mindfulness practices, meditation, or hobbies to protect your overall mental health.",
        metadata={"context": "human_health", "type": "genuine"}
    ),

    # --- 5 Out-of-Context Points (Using health keywords in unrelated fields) ---
    Document(
        page_content="To ensure the health of your server network, monitor the load stress and recover the database using a strict data diet plan.",
        metadata={"context": "IT_infrastructure", "type": "misleading_keywords"}
    ),
    Document(
        page_content="Exercise your stock options before the quarter ends to maintain a healthy financial portfolio and water down the tax implications.",
        metadata={"context": "finance", "type": "misleading_keywords"}
    ),
    Document(
        page_content="When the engine undergoes thermal stress, make sure the water pump is functioning to prevent the mechanical muscles of the piston from failing.",
        metadata={"context": "automotive_engineering", "type": "misleading_keywords"}
    ),
    Document(
        page_content="A balanced soil diet requires resting the agricultural land for a season so the earth can sleep and recover its natural nutrients.",
        metadata={"context": "agriculture", "type": "misleading_keywords"}
    ),
    Document(
        page_content="The structural health of the building depends on concrete hydration; adding too much water limits the tensile muscle of the foundation.",
        metadata={"context": "construction", "type": "misleading_keywords"}
    )
]

from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',model_kwargs={"device":0})


vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':0}),
)


mqretriever=MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k":5, 'lambda_mult':0.5},search_type='mmr'),
    llm=model,
)

query="How to stay healthy ?"

r1=mqretriever.invoke(query)
for i , docs in enumerate(r1):
    print(docs.page_content)
print("-"*100)

r2=vectorstore.similarity_search(query)
for i , docs in enumerate(r2):
    print(docs.page_content)