from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms.gpt4all import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.indexes import VectorstoreIndexCreator

PROMPT = 'SAOがクリアされた後アスナはどうなった？'

#template = """Answer the question based only on the following context:
#{context}
#
#Question: {question}
#
#Answer: Let's think step by step
#"""

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: Japanese
"""


# Document Loaders
#loader = DirectoryLoader(path="data", loader_cls=CSVLoader, glob='*.csv')
loader = DirectoryLoader(path="data", loader_cls=TextLoader, glob='sao.txt')
raw_docs = loader.load()

# Document Transformers
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 0)
docs = text_splitter.split_documents(raw_docs)

# Embedding
index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings()).from_loaders([loader])
#embeddings = HuggingFaceEmbeddings()
#faiss_db = FAISS.from_documents(documents=docs, embedding=embeddings)

# Vector Store
#faiss_db.save_local(FAISS_DB_PATH + FAISS_DB_DIR)

# RAG
results = index.vectorstore.similarity_search(PROMPT, k=3)
context = "\n".join([document.page_content for document in results])
#print(f"{context}")
#embedding_vector = embeddings.embed_query(PROMPT)
#contexts = faiss_db.similarity_search_by_vector(embedding_vector)
#context = "\n".join([document.page_content for document in contexts])
callbacks = [StreamingStdOutCallbackHandler()]

llm = GPT4All(model='models/ELYZA-japanese-Llama-2-13b-fast-instruct-q4_K_M.gguf', callbacks=callbacks, verbose=True)
#llm = GPT4All(model='models/ELYZA-japanese-Llama-2-13b-fast-instruct-q2_K.gguf', callbacks=callbacks, verbose=True)
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run(PROMPT)

