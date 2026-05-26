from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader=DirectoryLoader(
    path="Data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

data=loader.lazy_load()

