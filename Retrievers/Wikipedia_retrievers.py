from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(
    lang="en",
    top_k_results=2
)

query="Detail insights over the Indian Diplomacy"
result=retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"result {i+1}")
    print(doc.page_content)