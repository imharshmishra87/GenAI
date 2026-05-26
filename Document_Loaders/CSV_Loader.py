from langchain_community.document_loaders import CSVLoader

loader=CSVLoader('bengaluru_house_prices.csv')

data=loader.load()

print(data[0].page_content)
print(len(data))