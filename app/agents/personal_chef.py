from app.common.logger import logger
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage,HumanMessage,AIMessageChunk
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import sqlite3
from langchain.agents import create_agent, middleware
from langchain.tools import tool
from app.RAG.rewrite import rag_search
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware
# 加载环境变量
load_dotenv()
model=init_chat_model(model="deepseek-flash",
                      model_provider="openai",
                      api_key=os.getenv("DEEPSEEK_API_KEY"),
                      base_url=os.getenv("DEEPSEEK_BASE_URL"))
connect=sqlite3.connect("db/personal_chef.db",check_same_thread=False)
checkpoint=SqliteSaver(connect)
checkpoint.setup()
search_tool=TavilySearch(
    max_results=3,
    topic="general"
)
SUMMARIZER=SummarizationMiddleware(
            model="deepseek-chat",
            trigger=("fraction",0.8),
            keep=("fraction",0.2))
@tool()
def similarity_search(query:str,top_k:int=3):
    """在本地知识库中检索与用户问题相关的菜谱和做法
    Args:
    query: 用户的搜索需求或关键词
    top_k:返回菜谱的数量，默认3条
    Returns:
        检索到的菜谱列表"""
    logger.info(f"工具调用：similarity_search")
    try:
        recipes=rag_search(query,top_k=top_k)
        if not recipes:
            return "本地知识库没有找到相关菜谱，建议使用web_search搜索互联网"
    except Exception as e:
        logger.error("检索工具调用失败")
        return "工具调用失败"
    return recipes
@tool()
def web_search(query:str):
    """search the web for information"""
    return search_tool.invoke(query)
system_prompt="""
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用similarity_search工具检索本地知识库的菜谱，web_search 工具联网搜索食谱，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用similarity_search工具检索本地知识库的菜谱，如果知识库检索出来的菜谱与用户提供的食材关联不大或菜谱数量不足再调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""
agent=create_agent(model=model,
                   system_prompt=system_prompt,
                   tools=[web_search,similarity_search],
                   checkpointer=checkpoint,
                   middleware=[SUMMARIZER],
                   )
async def search_recipes(prompt:str,image:str,thread_id:str):
    logger.info(f"[用户]:{prompt}，“图片:{image}，thread_id:{thread_id}")
    if image and image.strip()!="":
        message=HumanMessage(content=[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":image}}
        ])
    else:
        message=HumanMessage(content=prompt)
    for chunk,metadata in agent.stream(
    {"messages":[message]},
    {"configurable":{"thread_id":thread_id}},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "model":
            continue
        if isinstance(chunk,AIMessageChunk) and chunk.content:
            yield chunk.content
def clear_messages(thread_id:str):
    logger.info(f"清空历史消息，对话:{thread_id}")
    checkpoint.delete_thread(thread_id)
def getmessages(thread_id:str)->list[dict[str, str]]:
    cp=checkpoint.get({"configurable":{"thread_id":thread_id}})
    if not cp:
        return []
    channel_values= cp.get("channel_values")
    if not channel_values:
        return []
    messages=channel_values.get("messages",[])
    if not messages:
        return []
    result=[]
    for msg in messages:
        if not msg.content:
            continue
        if isinstance(msg,AIMessage):
            result.append({"role":"assistant","content":msg.content})
        if isinstance(msg,HumanMessage):
            result.append({"role":"user","content":msg.content})
    return result
