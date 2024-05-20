from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# 資料の格納場所（ディレクトリ）
data_dir = "/tmp"

# ディレクトリの読み込み
loader = DirectoryLoader(data_dir, glob="*.pdf")

# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(
    model_name="./models/intfloat_multilingual-e5-large"
)

# テキストをチャンクに分割
split_texts = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,# 分割したチャンクごとの文字数
        chunk_overlap=50 # チャンク間で被らせる文字数
    )
)

