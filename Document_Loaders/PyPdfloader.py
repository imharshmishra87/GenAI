from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("a-practical-guide-to-building-agents.pdf")

data=loader.load()

print(len(data))
print(data[0].page_content)
print(data[1].metadata)