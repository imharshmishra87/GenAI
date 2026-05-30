from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text="hello harsh"

result=model.embed_query(text)
print(result)