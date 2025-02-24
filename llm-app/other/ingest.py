from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

def dictToVectorStore(input_dict):
    
    documents = [
        Document(page_content=item["content"], metadata={"title": item["title"]})
        for item in input_dict
    ]

    embeddings = OpenAIEmbeddings()

    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    ).from_documents(documents)

    index.vectorstore.save_local("faiss")