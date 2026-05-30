# 🤖 GenAI — LangChain & Agentic AI Systems

> A comprehensive collection of codes, functions, modules, and resources for building AI agentic systems using LangChain and Python.

---

## 📌 Overview

This repository is a structured, hands-on reference for developers learning or building with **Generative AI** — specifically using the **LangChain** ecosystem. It covers the full pipeline from loading raw documents to deploying production-ready agentic applications. Each directory targets a distinct component of a GenAI system, making it easy to study, reuse, and extend individual building blocks.

**Tech Stack:**
- 🐍 Python (63.6%) + Jupyter Notebooks (36.4%)
- 🦜 LangChain & LangChain Community
- 🤗 HuggingFace Transformers & Embeddings
- 🔗 OpenAI / Groq / Google Generative AI APIs
- 🗄️ FAISS, Chroma — Vector Stores
- 📄 PyPDF, Docx2txt, WebBaseLoader — Document Loaders

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/imharshmishra87/GenAI.git
cd GenAI

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Environment Variables

Most modules require API keys. Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## 📁 Repository Structure

```
GenAI/
├── Chains/
├── Document_Loaders/
├── Langchain_Models/
├── Langchain_Portfolio_Project/
├── Message_Templates/
├── Output_Parsers/
├── Retrievers/
├── Runnables/
├── Structured_outputs/
├── text-splitters/
├── tools/
├── vectorstore/
├── requirements.txt
└── .gitignore
```

---

## 📂 Directory Explanations

---

### 🔗 `Chains/`

**Purpose:** Demonstrates how to compose multiple LangChain components into sequential or conditional pipelines called *chains*.

A **Chain** connects components — a prompt template, an LLM, and an output parser — so that the output of one step flows into the next automatically. This is the backbone of most LangChain applications.

**Key Concepts Covered:**
- `LLMChain` — the simplest chain: prompt → LLM → output
- `SequentialChain` — chains where one chain's output feeds the next
- `RetrievalQA Chain` — combines a retriever with a language model to answer questions from documents
- `ConversationalRetrievalChain` — adds conversation memory to QA chains

**Typical Use Case:** Building a Q&A bot that retrieves relevant documents and synthesizes an answer with an LLM.

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="Tell me a joke about {topic}", input_variables=["topic"])
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"topic": "AI"})
```

---

### 📄 `Document_Loaders/`

**Purpose:** Shows how to ingest data from various file formats and sources into LangChain `Document` objects.

Before an AI system can reason over your data, it must first load and parse it. LangChain provides a rich set of **Document Loaders** that extract text from different source types.

**Sources Covered:**
| Loader | Source Type |
|--------|-------------|
| `PyPDFLoader` | PDF files |
| `Docx2txtLoader` | Microsoft Word (.docx) |
| `WebBaseLoader` | Web pages / URLs |
| `CSVLoader` | CSV/tabular data |
| `TextLoader` | Plain text files |
| `UnstructuredFileLoader` | Mixed/unstructured content |

**Key Concepts Covered:**
- Loading single and multiple documents
- Handling metadata (source, page number, etc.)
- Directory-level loading with `DirectoryLoader`
- Lazy loading for large datasets

**Typical Use Case:** Loading a set of PDFs from a folder before splitting and indexing them into a vector store.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
documents = loader.load()
print(documents[0].page_content)
```

---

### 🧠 `Langchain_Models/`

**Purpose:** Explores the different types of language model integrations available in LangChain.

LangChain abstracts over multiple LLM providers through a consistent interface. This directory demonstrates how to connect to and configure various models.

**Model Types Covered:**
- **LLMs** — Standard completion models (text-in, text-out)
- **Chat Models** — Dialogue-oriented models using message roles (system, human, AI)
- **Embedding Models** — Convert text into numerical vectors for similarity search

**Providers Demonstrated:**
- `ChatOpenAI` — OpenAI GPT-3.5 / GPT-4
- `ChatGroq` — Groq-hosted open-source models (Llama, Mixtral)
- `ChatGoogleGenerativeAI` — Google Gemini models
- `HuggingFaceHub` — Open-source models via HuggingFace
- `OllamaLLM` — Locally hosted models via Ollama

**Key Concepts Covered:**
- Switching between providers with minimal code change
- Configuring temperature, max tokens, streaming
- Understanding the difference between `invoke()`, `stream()`, and `batch()`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke("Explain quantum computing in simple terms.")
print(response.content)
```

---

### 🗂️ `Langchain_Portfolio_Project/`

**Purpose:** A complete, end-to-end portfolio project that demonstrates how all the individual components come together into a real GenAI application.

This is the capstone section of the repository. It integrates document loaders, text splitters, vector stores, retrievers, chains, and output parsers into a cohesive deployable application.

**What It Likely Includes:**
- A **RAG (Retrieval-Augmented Generation)** pipeline
- Ingestion phase: load → split → embed → store
- Query phase: retrieve → prompt → generate → parse output
- Possibly a simple **Streamlit** or **FastAPI** interface

**Key Concepts Covered:**
- End-to-end system design for GenAI applications
- Separation of concerns between data ingestion and inference
- Practical patterns for production-readiness

**Why It Matters:** This project demonstrates real-world applicability — not just isolated components but a working system you can present in interviews or deploy as a proof of concept.

---

### 💬 `Message_Templates/`

**Purpose:** Covers the use of **Prompt Templates** and message formatting patterns in LangChain.

The quality of an LLM's response depends heavily on how the prompt is structured. This directory focuses on the art and science of prompt engineering via LangChain's templating system.

**Template Types Covered:**
- `PromptTemplate` — Simple string templates with variable substitution
- `ChatPromptTemplate` — Structured multi-turn message templates
- `SystemMessagePromptTemplate` — For setting model persona/behavior
- `HumanMessagePromptTemplate` — For user messages
- `MessagesPlaceholder` — For injecting conversation history

**Key Concepts Covered:**
- Defining reusable, parameterized prompts
- Few-shot prompting (providing examples in the prompt)
- System vs human vs AI message roles
- Dynamic prompt construction

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks like a pirate."),
    ("human", "{user_input}")
])
formatted = prompt.format_messages(user_input="What is machine learning?")
```

---

### 🖨️ `Output_Parsers/`

**Purpose:** Demonstrates how to parse, structure, and validate raw LLM text responses into usable Python objects.

LLMs output free-form text. **Output Parsers** convert that text into structured data formats that your application can work with reliably.

**Parsers Covered:**
| Parser | Output Format |
|--------|---------------|
| `StrOutputParser` | Raw string |
| `JsonOutputParser` | Python dictionary/JSON |
| `CommaSeparatedListOutputParser` | Python list |
| `PydanticOutputParser` | Validated Pydantic model |
| `StructuredOutputParser` | Custom response schema |

**Key Concepts Covered:**
- Attaching parsers to chains with the `|` pipe operator
- Using format instructions to guide LLM output
- Pydantic-based validation for robust parsing
- Error handling when output doesn't match expected format

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse("apple, banana, cherry")
# Output: ['apple', 'banana', 'cherry']
```

---

### 🔍 `Retrievers/`

**Purpose:** Explores the various strategies for retrieving relevant documents from a knowledge base in response to a user query.

A **Retriever** is the component that fetches the most relevant information from a vector store or other data source. This is the "R" in **RAG** (Retrieval-Augmented Generation).

**Retriever Types Covered:**
- `VectorStoreRetriever` — Standard similarity search over embeddings
- `MultiQueryRetriever` — Generates multiple query variants to improve recall
- `ContextualCompressionRetriever` — Compresses retrieved docs to only the relevant portions
- `EnsembleRetriever` — Combines results from multiple retrievers
- `SelfQueryRetriever` — Converts natural language into structured metadata filters

**Key Concepts Covered:**
- `similarity_search` vs `mmr` (Maximal Marginal Relevance) — balancing relevance and diversity
- Setting `k` (number of results to return)
- Filtering by metadata
- Re-ranking retrieved results

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
docs = retriever.invoke("What is transformer architecture?")
```

---

### ⚙️ `Runnables/`

**Purpose:** Introduces the **LangChain Expression Language (LCEL)** and the `Runnable` interface — the modern, composable way to build LangChain pipelines.

`Runnables` are the core abstraction in modern LangChain. Every component (prompt, model, parser, retriever) is a Runnable, meaning they all share a common interface (`invoke`, `stream`, `batch`) and can be composed with the `|` pipe operator.

**Key Runnables Covered:**
- `RunnablePassthrough` — Passes input unchanged (used to inject context)
- `RunnableParallel` — Runs multiple runnables in parallel and merges results
- `RunnableLambda` — Wraps any Python function as a Runnable
- `RunnableWithMessageHistory` — Adds stateful conversation memory
- `RunnableBranch` — Conditional routing between different chains

**Key Concepts Covered:**
- Building chains with the `|` (pipe) operator — LCEL syntax
- Parallel execution for efficiency
- Streaming responses token by token
- Batching multiple inputs for throughput

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
chain.invoke("How does attention work in transformers?")
```

---

### 📐 `Structured_outputs/`

**Purpose:** Covers how to force language models to return responses in a specific, structured schema using LangChain's structured output capabilities.

When building applications, you often need the LLM to return well-defined data (e.g., a JSON object with specific fields) rather than free text. This directory explores how to enforce that.

**Approaches Covered:**
- `with_structured_output()` — The modern, preferred method using function calling / tool use under the hood
- Pydantic models as schema definitions
- JSON schema-based output enforcement
- `TypedDict` for lightweight schema definitions

**Key Concepts Covered:**
- Defining output schemas with Pydantic `BaseModel`
- Combining structured output with validation
- Using `Optional` fields for flexible schemas
- Error handling when the model returns malformed output

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(PersonInfo)
result = structured_llm.invoke("John is a 30-year-old software engineer.")
print(result.name, result.age)  # John 30
```

---

### ✂️ `text-splitters/`

**Purpose:** Demonstrates how to split large documents into smaller, manageable chunks before embedding them into a vector store.

LLMs have a context window limit, and embedding models work best on focused text chunks. **Text Splitters** intelligently divide documents while preserving semantic coherence.

**Splitters Covered:**
| Splitter | Strategy |
|----------|----------|
| `RecursiveCharacterTextSplitter` | Splits by paragraphs → sentences → words (recommended) |
| `CharacterTextSplitter` | Splits on a single separator character |
| `TokenTextSplitter` | Splits based on token count (model-aware) |
| `MarkdownHeaderTextSplitter` | Splits by Markdown heading structure |
| `HTMLHeaderTextSplitter` | Splits HTML by heading tags |
| `SemanticChunker` | Splits based on semantic similarity between sentences |

**Key Concepts Covered:**
- `chunk_size` — Maximum size of each chunk
- `chunk_overlap` — How many characters/tokens overlap between chunks (prevents context loss at boundaries)
- Choosing the right splitter for different document types
- Inspecting chunks before embedding

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
```

---

### 🛠️ `tools/`

**Purpose:** Explores how to give LangChain agents the ability to take actions by integrating external **Tools**.

A **Tool** is any function or API an AI agent can call to interact with the outside world — search the web, run code, query a database, check the weather, etc. This is what makes LLMs genuinely *agentic*.

**Tools Demonstrated:**
- `DuckDuckGoSearchRun` — Web search without API key
- `WikipediaQueryRun` — Query Wikipedia articles
- `PythonREPLTool` — Execute Python code
- `ArxivQueryRun` — Search academic papers
- Custom tools built with `@tool` decorator

**Key Concepts Covered:**
- Defining custom tools with `@tool` decorator and type hints
- Tool descriptions — critical for the agent to know when/how to use a tool
- Building tool-calling agents (`create_tool_calling_agent`)
- `AgentExecutor` — The runtime loop that manages tool calls
- ReAct pattern: Reason → Act → Observe → Repeat

```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)
```

---

### 🗄️ `vectorstore/`

**Purpose:** Covers how to store, manage, and query text embeddings using vector databases — the foundation of semantic search and RAG systems.

A **Vector Store** is a specialized database that stores text as high-dimensional numerical vectors (embeddings). Queries are answered by finding the vectors most *similar* (not identical) to the query embedding — enabling semantic, meaning-based search.

**Vector Stores Covered:**
- `FAISS` — Facebook AI Similarity Search; fast, local, in-memory (great for prototyping)
- `Chroma` — Lightweight, open-source, persistent local vector DB
- Potentially `Pinecone` or `Weaviate` — Managed cloud vector databases

**Embedding Models Used:**
- `OpenAIEmbeddings` — `text-embedding-ada-002` / `text-embedding-3-small`
- `HuggingFaceEmbeddings` — Open-source embedding models (e.g., `sentence-transformers`)
- `GoogleGenerativeAIEmbeddings` — Gemini embedding models

**Key Concepts Covered:**
- Creating a vector store from documents
- Persisting a vector store to disk and reloading it
- Performing similarity search with scores
- Adding and deleting documents from the store
- Connecting a vector store to a retriever

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 🧩 How the Components Fit Together

The directories in this repository represent the **complete data pipeline** for a RAG-based GenAI application:

```
Raw Data
   │
   ▼
Document_Loaders     ← Load PDFs, Word docs, web pages, CSVs
   │
   ▼
text-splitters       ← Break large documents into smaller chunks
   │
   ▼
Langchain_Models     ← Generate embeddings (embedding models)
   │
   ▼
vectorstore          ← Store chunks as searchable vectors
   │
   ▼
Retrievers           ← Find the most relevant chunks for a query
   │
   ▼
Message_Templates    ← Format the query + retrieved context into a prompt
   │
   ▼
Langchain_Models     ← Send prompt to LLM (chat models)
   │
   ▼
Output_Parsers       ← Parse LLM response into structured data
   │
   ▼
Chains / Runnables   ← Orchestrate all above steps into a single pipeline
   │
   ▼
tools                ← Give agents external capabilities (search, code, APIs)
   │
   ▼
Langchain_Portfolio_Project  ← Full working application
```

---

## 📦 Dependencies

Key packages from `requirements.txt`:

```
langchain
langchain-community
langchain-openai
langchain-groq
langchain-google-genai
langchain-huggingface
faiss-cpu
chromadb
pypdf
docx2txt
sentence-transformers
python-dotenv
streamlit          # (for UI demos)
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! To add a new example or improve existing ones:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-new-loader`)
3. Add your code with clear comments and a brief docstring
4. Submit a pull request with a description of what you've added

---

## 📄 License

This project is open-source. Please refer to the repository for license details.

---

## 👤 Author

**Harsh Mishra**
- GitHub: [@imharshmishra87](https://github.com/imharshmishra87)

---

> ⭐ If this repository helped you understand LangChain and GenAI systems, please consider giving it a star!
