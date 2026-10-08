from langchain_milvus import Milvus,BM25BuiltInFunction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter,RecursiveJsonSplitter
from app.RAG.embeddings import  get_embeddings
import os
import json
def get_vector_store(collection_name:str="langchain_collection"):
    vector_store=Milvus(
        get_embeddings(),
        collection_name=collection_name,
        builtin_function=BM25BuiltInFunction(
            analyzer_params={"type":"chinese"}
        ),
        vector_field=["dense","sparse"],
        connection_args={
            "uri":os.getenv("ZILI_URI"),
            "token":os.getenv("ZILI_TOKEN")
        },
        drop_old=False
    )
    return vector_store
def load_recipes(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    documents = []
    ids=[]
    for recipe in recipes:
        ids.append(recipe["id"])
        # 把菜谱的关键信息拼成一段文本作为 page_content
        content = (f"菜名：{recipe['name']}\n"
            f"菜系：{recipe['cuisine']}\n"
            f"难度：{recipe['difficulty']}，烹饪时间：{recipe['cooking_time']}\n"
            f"食材：{', '.join(recipe['ingredients'])}\n"
            f"标签：{', '.join(recipe['tags'])}\n"
            f"步骤：{recipe['steps']}\n"
            f"简介：{recipe['description']}")
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "id": recipe["id"],
                    "name": recipe["name"],
                    "cuisine": recipe["cuisine"],
                    "tags": ",".join(recipe["tags"]),
                    "ingredients":",".join(recipe["ingredients"])
                }
            )
        )
    return [ids,documents]
"""docs = load_recipes("./resource/菜谱.json")
vector_stores=get_vector_store()
vector_stores.add_documents(docs[1],ids=docs[0])
print(docs)
query="鸡肉"
result=vector_stores.similarity_search_with_score(
    query,
    k=3,
    ranker_type="rrf",
    ranker_params={"k":60}
)"""
"""vectorstore.similarity_search(
    "What is CoT?",
    k=1,
    expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
)

# The same as:
# vectorstore.as_retriever(search_kwargs=dict(
#     k=1,
#     expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
# )).invoke("What is CoT?")"""
def reset_vector():
    try:
        vector_store=get_vector_store()
        vector_store.drop()
        return "删除collection完成"
    except:
        return "删除失败"