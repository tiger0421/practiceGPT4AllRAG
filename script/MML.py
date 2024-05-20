import pandas as pd
import re
from datasets import load_dataset

from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

INPUT_DATA_PATH = "./data/data.txt"
question = "Who is Kazuto's daughter?"
llm_path = './models/gpt4all-falcon-q4_0.gguf'

loader = TextLoader(INPUT_DATA_PATH, encoding="utf-8")
index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings()).from_loaders([loader])


callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=llm_path, callbacks=callbacks, verbose=True, backend='gptj')

results = index.vectorstore.similarity_search(question, k=4)
context = "\n".join([document.page_content for document in results])
print(f"{context}")


template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

llm_chain = LLMChain(prompt=prompt, llm=llm)
print("### GPT4ALL answer ###")
print(llm_chain.run(question))

