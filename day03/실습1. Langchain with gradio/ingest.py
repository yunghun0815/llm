
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

#VetcorDB로 변환
def pdfToVectorStore(file):
    #1. PDF Load
    loader = PyPDFLoader(file)
    documents = loader.load()

    #2 Vector Embedding
    embeddings = OpenAIEmbeddings()

    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        ).from_loaders([loader])

    index.vectorstore.save_local("faiss")