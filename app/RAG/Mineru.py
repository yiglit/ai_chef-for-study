from langchain_mineru import MinerULoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.documents import Document
from vector_store import get_vector_store
import os
load_dotenv()
def load_docs(path:str="./resource/浙江风味2.pdf")->list[Document]:
    loader=MinerULoader(source=path,
                        mode="precision",
                        token=os.getenv("MINERU_TOKEN"))
    docs=loader.load()
    return docs
recursive_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。"]
)
chunks=recursive_splitter.split_documents(load_docs())
vector_store=get_vector_store()
ids=vector_store.add_documents(chunks)
