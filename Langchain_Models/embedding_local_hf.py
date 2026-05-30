from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':0})
text="Capital of india is france"
documents=["harsh is a great man","Harsh loves icecream","Mango is mine favourite"]

result=model.embed_documents(documents)
print(str(result))