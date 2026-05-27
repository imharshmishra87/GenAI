from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


docs = [
    Document(
        page_content="Langchain is a specialized data framework designed to efficiently ingest, structure, index, and retrieve custom private data for Large Language Models.",
        metadata={"framework": "Transformers"}
    ),
    Document(
        page_content="Haystack is an open-source framework by deepset that excels at building production-ready search systems, RAG pipelines, and conversational AI.",
        metadata={"framework": "Haystack"}
    ),
    Document(
        page_content="Semantic Kernel is Microsoft's orchestration SDK that seamlessly integrates AI models with conventional programming languages and enterprise plugins.",
        metadata={"framework": "Semantic Kernel"}
    ),
    Document(
        page_content="AutoGen is a powerful multi-agent framework that enables the development of complex LLM applications using AI personas that converse and collaborate with each other.",
        metadata={"framework": "AutoGen"}
    ),
    Document(
        page_content="Hugging Face Transformers provides accessible, state-of-the-art pre-trained models and APIs for building highly customized natural language processing pipelines.",
        metadata={"framework": "Transformers"}
    ),
    Document(
        page_content="LangChain is a powerful open-source framework designed to simplify the creation of applications powered by Large Language Models (LLMs)",
        metadata={"framework": "Transformers"}
    )
]

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':0}),
    collection_name="second_table"
)

retriever=vectorstore.as_retriever(search_kwargs={'k':2, 'lambda_mult':0.5}, search_type='mmr')

query="what is langchain ?"

result=retriever.invoke(query)
for i, doc in enumerate(result):
    print(doc.page_content)