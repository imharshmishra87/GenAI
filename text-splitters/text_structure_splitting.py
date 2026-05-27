from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""Where to Find AI Datasets for NLP and Computer Vision
When it comes to finding AI datasets for NLP and computer vision, several platforms are offering both free and paid options. Here are some of the top resources you can explore:

Best free platforms for computer vision and NLP datasets
Humans in the Loop-Free datasets
Humans in the Loop offers high-quality, free dataset collection for AI applications, including NLP and computer vision. Whether you’re looking for datasets for training image classifiers or language models, you’ll find a valuable resource here.

Unlike many free datasets without proper selection, Humans in the Loop (HITL) free datasets provide a highly-curated alternative. 

As discussed in this article, high-quality datasets are often available as paid options. The advantage of HITL is that it provides high-quality free datasets, with human annotators continuously reviewing them.

The HITL-free datasets cover a wide range of industries:
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks=splitter.split_text(text)
print(chunks)