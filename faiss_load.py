from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# インデックスのパス
index_path = "./storage"

# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# インデックスの読み込み
index = FAISS.load_local(
    folder_path=index_path,
    embeddings=embedding_model
)

question = "ゲームマスターは誰"
embedding_vector = embedding_model.embed_query(question)

#docs = index.similarity_search(question, k = 4)
docs = index.similarity_search_by_vector(embedding_vector, k = 4)
context = "\n".join([document.page_content for document in docs])

template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

print(prompt.format(context = context, question = question))

