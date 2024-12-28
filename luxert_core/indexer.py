from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
import chromadb
from pathlib import Path

class Indexer:
    def __init__(self, persist_dir, collection_name="luxert-em", embedding_model_name="all-MiniLM-L6-v2"):
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name

        embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name, 
            model_kwargs={"trust_remote_code": True}
        )

        persistent_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = persistent_client.get_or_create_collection(self._collection_name)
        self._vectordb = Chroma(
            client=persistent_client,
            collection_name=self._collection_name,
            embedding_function=embeddings,
        )

    def index_files(self, text_files_path):
        text_files_path = Path(text_files_path)
        for file in text_files_path.iterdir():
            if file.is_file():
                with open(file, "r") as f:
                    document = Document(page_content=f.read())
                    self.index_document(document)

    def index_document(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128
        )
        split_page_content = text_splitter.split_text(document.page_content)
        for segment in split_page_content:
            segment_hash = hashlib.md5(segment.encode()).hexdigest()
            if not self._collection.get(ids=segment_hash).get('ids'):
                self._collection.add(ids=[segment_hash], documents=[segment])
            else:
                print(f"Document with hash {segment_hash} already exists in collection. Skipping...")

    def index_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128
        )
        documents = text_splitter.split_documents(documents)
        self._vectordb.add_documents(documents)

    def get_all_documents(self):
        return self._collection.get().get('documents')

    def get_retriever(self):
        return self._vectordb.as_retriever()
    