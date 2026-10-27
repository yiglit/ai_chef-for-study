"""MinerU 文档入库流水线：装载 PDF -> Markdown -> 切分 -> 写入 Milvus 向量库。"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mineru import MinerULoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.RAG.vector_store import get_vector_store
from app.common.logger import logger
RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
RESOURCE_DIR = RAG_DIR / "resource"
OUTPUT_DIR = RESOURCE_DIR / "output"
DEFAULT_SOURCE = RESOURCE_DIR / "浙江风味2.pdf"
DEFAULT_COLLECTION = "mineru_docs"
load_dotenv(PROJECT_ROOT / ".env")
def load_documents(
    source: str | Path,
    *,
    mode: str = "precision",
    ocr: bool = True,
    split_pages: bool = False,
) -> list[Document]:
    """用 MinerU 解析 PDF，返回 Markdown 形式的 Document 列表。"""
    loader = MinerULoader(
        source=str(source),
        mode=mode,
        ocr=ocr,
        split_pages=split_pages,
    )
    documents = loader.load()
    logger.info(f"MinerU 解析完成：{source}，共 {len(documents)} 个 Document")
    return documents
def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """按 Markdown 结构切分文档，并把元数据统一成 Milvus 可写入的固定字段。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ", "\n## ", "\n### ", "\n#### ",
            "\n\n", "\n", "。", "！", "？", "；", "，", "",
        ],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source") or "")
        page = chunk.metadata.get("page")
        # Milvus 不支持 None，且同一 collection 的字段必须保持一致
        chunk.metadata = {
            "source": source,
            "filename": chunk.metadata.get("filename") or Path(source).name,
            "page": int(page) if page is not None else 0,
            "chunk_index": index,
            "loader": "mineru",
        }

    logger.info(f"文档切分完成，共 {len(chunks)} 个切分块")
    return chunks


def ingest_document(
    source: str | Path = DEFAULT_SOURCE,
    collection_name: str = DEFAULT_COLLECTION,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    mode: str = "precision",
    ocr: bool = True,
    split_pages: bool = False,
    save_markdown: bool = True,
) -> int:
    """完整流程：解析 -> 保存 Markdown -> 切分 -> 写入向量库。"""
    source = Path(source).expanduser()

    documents = load_documents(source, mode=mode, ocr=ocr, split_pages=split_pages)

    if save_markdown:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        markdown_path = OUTPUT_DIR / f"{source.stem}.md"
        markdown_path.write_text(
            "\n\n".join(doc.page_content for doc in documents),
            encoding="utf-8",
        )
        logger.info(f"Markdown 已保存：{markdown_path}")

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    ids = [f"{source.stem}-{index:04d}" for index in range(len(chunks))]

    vector_store = get_vector_store(collection_name)
    if vector_store.client.has_collection(collection_name):
        # 主键固定，重复入库时覆盖而不是产生重复数据
        vector_store.upsert(ids=ids, documents=chunks)
    else:
        vector_store.add_documents(documents=chunks, ids=ids)

    logger.info(
        f"入库完成：collection={collection_name}，共写入 {len(chunks)} 个切分块"
    )
    return len(chunks)


if __name__ == "__main__":
    import sys

    ingest_document(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE)
