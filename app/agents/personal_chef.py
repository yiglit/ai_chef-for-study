from app.common.logger import logger
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage,HumanMessage,AIMessageChunk
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import sqlite3
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
# 加载环境变量
load_dotenv()
model=init_chat_model(model="deepseek-v4-flash-vision-exp",
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

@tool()
def web_search(query:str):
    """search the web for information"""
    return search_tool.invoke(query)
system_prompt="""
#角色：你是一位ai大厨，需要根据用户提供的食材为用户提供可行的菜品
#目标：举出3道菜品并提供菜品的制作流程
#能力与要求：如果用户提供的是图片，需要先解析用户图片里的全部食材，列出食材清单，分析色泽与新鲜度，列出可用的食材清单，如果是文字，直接对文字进行分析列出清单，接着再优先调用web_search工具联网搜索食材清单上的食材可做的菜谱和参考图，
根据制作难度与营养价值对菜品进行打分排列，营养价值高，制作难度低的排在前面。输出结构化数据：包含食谱信息，对菜品的评分，推荐理由，以及一张菜品的参考图。
#限制：优先利用web_search工具进行菜品的搜索，如果搜索不到再利用模型自己发挥想象。
"""
agent=create_agent(model=model,
                   system_prompt=system_prompt,
                   tools=[web_search],
                   checkpointer=checkpoint
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
        if isinstance(msg.content,AIMessage):
            result.append({"role":"assistant","content":msg.content})
        if isinstance(msg.content,HumanMessage):
            result.append({"role":"user","content":msg.content})
    return result