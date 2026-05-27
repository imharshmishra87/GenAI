from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':0})

splitter=SemanticChunker(
    model, 
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

text = (
    "Neural networks are excellent at image recognition. They process pixels through layers. "
    "On a completely different note, baking a sourdough bread requires active wild yeast culture. "
    "You must maintain a strict feeding schedule for your starter culture."
)

data=splitter.split_text(text)
print(data)