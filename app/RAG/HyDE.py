from langchain.chat_models import  init_chat_model
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from app.common.logger import logger
from app.RAG.vector_store import get_vector_store
load_dotenv()
_model = None
def get_model():
    global _model
    if _model is None:
        model=init_chat_model(model="deepseek-flash",
                            model_provider="openai",
                            api_key=os.getenv("DEEPSEEK_API_KEY"),
                            base_url=os.getenv("DEEPSEEK_BASE_URL"))
    return model
def rewrite_query(query:str)->dict:
    try:
        model=get_model()
        messages=[
            SystemMessage(content=(
                "你是一个菜谱搜索 Query 改写助手。用户会用口语化的方式描述想吃什么菜，"
                "你需要将其改写为适合向量检索的中文关键词串。\n"
                "要求：\n"
                "1. 提取核心食材、烹饪方式、口味偏好、菜系等检索维度\n"
                "2. 输出 5~10 个关键词，用空格分隔\n"
                "3. 只输出关键词，不要解释\n"
                "示例：\n"
                "用户：今天天好热想吃点清淡的 → 清淡 凉菜 清蒸 夏日 低卡 凉拌\n"
                "用户：冰箱里有鸡肉和土豆怎么做 → 鸡肉 土豆 家常菜 炖菜 炒菜 红烧\n"
                "用户：减肥期间能吃什么肉 → 减脂 低卡 高蛋白 鸡胸肉 鱼肉 清蒸 水煮"
                )),
            HumanMessage(content=f"照系统提示词改写以下内容：{query}")
            ]
        response=model.invoke(messages)
        rewritten_query=response.content.strip()
        keywords=rewritten_query.split()
        logger.info(f"RAG Query改写:{query}->{rewritten_query},keywords:{keywords}")
        return {"original": query, "rewritten_query": rewritten_query, "keywords": keywords}
    except Exception as e:
        logger.error("问题改写失败")
        return {"original": query, "rewritten_query": query, "keywords": query}
#向量召回，语义检索
def vector_call(query:str,top_k:int=3)->list:
    try:
        collection=get_vector_store()
        results=collection.similarity_search(
            query=query,
            k=top_k,
            ranker_type="rrf",
            ranker_params={"k":60}
            )
        if not results:
            return []
        recipes=[]
        for i in results:
            recipes.append(i.page_content)
        logger.info(f"向量召回{len(recipes)}")
        return recipes
    except Exception as e:
        logger.error(f"向量召回失败")
        return []
def rag_search(query:str,top_k:int=3):
    query_info=rewrite_query(query)
    vectorize=vector_call(query_info["rewritten_query"],top_k)
    return vectorize
"""text=rag_search("豆腐")
for i,content in enumerate(text):
    print(f"第{i+1}道推荐\n{content}")"""