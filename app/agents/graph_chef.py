from typing import TypedDict,Annotated,Literal
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langgraph.constants import START,END
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,ToolMessage,AIMessage,AIMessageChunk,RemoveMessage
from langgraph.graph.message import add_messages
import os
from langgraph.store.sqlite.base import SqliteIndexConfig
from langmem import create_memory_store_manager,ReflectionExecutor
from langgraph.prebuilt import ToolNode,tools_condition
from langchain.tools import tool
from langchain_tavily import TavilySearch
from app.RAG.embeddings import get_embeddings
from app.RAG.rewrite import rag_search
from app.common.logger import logger
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
import sqlite3
import asyncio
import threading
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter
from pydantic import BaseModel
load_dotenv()

class agentstate(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    summary:str
    intent:Literal["nutrition","recipe","general"]
    thread_id:str

@tool
def calculate_nutrition(ingredients: str, servings:int= 1) -> str:
    """
    计算食材的营养成分（卡路里、蛋白质、脂肪、碳水化合物等）。

    Args:
        ingredients: 食材清单字符串，格式如"鸡胸肉200g, 西兰花150g, 大米100g"
        servings: 份数，默认为1人份

    Returns:
        结构化的营养分析报告
    """
    logger.info(f"[工具调用] calculate_nutrition: ingredients={ingredients}, servings={servings}")

    nutrition_db = {
        "鸡胸肉": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0, "fiber": 0},
        "鸡腿肉": {"calories": 184, "protein": 20, "fat": 11, "carbs": 0, "fiber": 0},
        "猪肉": {"calories": 242, "protein": 27, "fat": 14, "carbs": 0, "fiber": 0},
        "牛肉": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0, "fiber": 0},
        "鸡蛋": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1, "fiber": 0},
        "豆腐": {"calories": 76, "protein": 8, "fat": 4.8, "carbs": 1.9, "fiber": 0.3},
        "大米": {"calories": 365, "protein": 7, "fat": 0.6, "carbs": 79, "fiber": 1.3},
        "面条": {"calories": 270, "protein": 9, "fat": 1.5, "carbs": 56, "fiber": 2.5},
        "西兰花": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7, "fiber": 2.6},
        "胡萝卜": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 10, "fiber": 2.8},
        "土豆": {"calories": 77, "protein": 2, "fat": 0.1, "carbs": 17, "fiber": 2.2},
        "番茄": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9, "fiber": 1.2},
        "黄瓜": {"calories": 15, "protein": 0.8, "fat": 0.1, "carbs": 3.6, "fiber": 0.5},
        "青菜": {"calories": 15, "protein": 1.5, "fat": 0.2, "carbs": 2.7, "fiber": 1.6},
        "菠菜": {"calories": 23, "protein": 2.9, "fat": 0.4, "carbs": 3.6, "fiber": 2.2},
        "白菜": {"calories": 12, "protein": 1.5, "fat": 0.1, "carbs": 2.2, "fiber": 1},
        "洋葱": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9.3, "fiber": 1.7},
        "大蒜": {"calories": 149, "protein": 6.4, "fat": 0.5, "carbs": 33, "fiber": 2.1},
        "辣椒": {"calories": 40, "protein": 1.9, "fat": 0.3, "carbs": 8.8, "fiber": 1.5},
        "青椒": {"calories": 20, "protein": 0.9, "fat": 0.2, "carbs": 4.6, "fiber": 1.7},
        "茄子": {"calories": 25, "protein": 1, "fat": 0.2, "carbs": 5.7, "fiber": 3},
        "玉米": {"calories": 86, "protein": 3.2, "fat": 1.2, "carbs": 19, "fiber": 2.7},
        "蘑菇": {"calories": 22, "protein": 3.1, "fat": 0.3, "carbs": 3.3, "fiber": 1},
        "虾": {"calories": 99, "protein": 24, "fat": 0.3, "carbs": 0.2, "fiber": 0},
        "鱼": {"calories": 165, "protein": 22, "fat": 8, "carbs": 0, "fiber": 0},
        "三文鱼": {"calories": 208, "protein": 20, "fat": 13, "carbs": 0, "fiber": 0},
        "牛奶": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5, "fiber": 0},
        "酸奶": {"calories": 59, "protein": 3.5, "fat": 3.3, "carbs": 3.6, "fiber": 0},
        "芝士": {"calories": 402, "protein": 25, "fat": 33, "carbs": 3.5, "fiber": 0},
        "苹果": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14, "fiber": 2.4},
        "香蕉": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23, "fiber": 2.6},
        "橄榄油": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0, "fiber": 0},
        "食用油": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0, "fiber": 0},
        "盐": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
        "酱油": {"calories": 53, "protein": 8, "fat": 0.1, "carbs": 4.9, "fiber": 0},
        "糖": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100, "fiber": 0},
    }
    ingredients_list=[x.strip() for x in ingredients.split(",")]
    parsed_ingredients=[]
    for item in ingredients_list:
        name = item
        weight = 100
        import re
        match = re.search(r'(\d+)\s*g', item)
        if match:
            weight = int(match.group(1))
            name = re.sub(r'\d+\s*g', '', item).strip()
        else:
            match = re.search(r'(\d+)\s*克', item)
            if match:
                weight = int(match.group(1))
                name = re.sub(r'\d+\s*克', '', item).strip()
        parsed_ingredients.append({"name": name, "weight": weight})
    results = []
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0}
    for ing in parsed_ingredients:
        match_key=None
        for key in nutrition_db:
            if key in ing["name"] or ing["name"] in key:
                match_key = key
                break
        if match_key:
            factor=ing["weight"]/100.0*servings
            nutrition = nutrition_db[match_key]
            item_result={
                "name":ing["name"],
                "weight":ing["weight"],
                "calories":round(nutrition["calories"]*factor,1),
                "protein": round(nutrition["protein"] * factor, 1),
                "fat": round(nutrition["fat"] * factor, 1),
                "carbs": round(nutrition["carbs"] * factor, 1),
                "fiber": round(nutrition["fiber"] * factor, 1),
            }
            results.append(item_result)
            for k in total:
                total[k]+=item_result[k]
        else:
            results.append({"name": ing["name"], "weight": ing["weight"], "note": "未找到该食材的营养数据"})
    output = []
    output.append(f"## 营养成分分析报告（{servings}人份）\n")
    output.append("### 各食材营养明细：\n")
    output.append("| 食材 | 重量 | 卡路里 | 蛋白质(g) | 脂肪(g) | 碳水(g) | 纤维(g) |")
    output.append("|------|------|--------|-----------|---------|---------|---------|")
    for r in results:
        if "note" in r:
            output.append(f"| {r['name']} | {r['weight']}g | - | - | - | - | - | *{r['note']}* |")
        else:
            output.append(
                f"| {r['name']} | {r['weight']}g | {r['calories']} | {r['protein']} | {r['fat']} | {r['carbs']} | {r['fiber']} |")

    output.append("\n### 营养汇总：\n")
    output.append(f"- **总卡路里**：{total['calories']} kcal")
    output.append(f"- **蛋白质**：{total['protein']} g")
    output.append(f"- **脂肪**：{total['fat']} g")
    output.append(f"- **碳水化合物**：{total['carbs']} g")
    output.append(f"- **膳食纤维**：{total['fiber']} g\n")

    output.append("### 营养建议：\n")
    calories_per_meal = total["calories"]
    if calories_per_meal < 300:
        output.append("- ⚠️ 本餐热量偏低，建议适当增加主食或蛋白质摄入")
    elif calories_per_meal > 800:
        output.append("- ⚠️ 本餐热量偏高，注意控制分量或增加运动量")
    else:
        output.append("- ✅ 本餐热量适中，适合正常成年人一餐")
    if total["protein"] < 15:
        output.append("- ⚠️ 蛋白质摄入不足，建议增加肉类、蛋类或豆制品")
    elif total["protein"] > 60:
        output.append("- ℹ️ 蛋白质摄入较高，肾功能不佳者请注意")
    else:
        output.append("- ✅ 蛋白质含量充足")
    return "\n".join(output)
@tool()
def similarity_search(query:str,top_n:int=3):
    """在本地知识库中检索与用户问题相关的菜谱和做法
    Args:
    query: 用户的搜索需求或关键词
    top_n:返回菜谱的数量，默认3条
    Returns:
        检索到的菜谱列表"""
    try:
        results=rag_search(query,top_n=top_n)
        if not results:
            return "本地知识库没有找到相关菜谱，建议使用web_search搜索互联网"
        recipes = "\n\n".join(i.page_content for i in results)
    except Exception as e:
        return "工具调用失败"
    return recipes

@tool()
def web_search(query:str):
    """search the web for information"""
    return search_tool.invoke(query)
LLM=init_chat_model(model="deepseek-flash",
                      model_provider="openai",
                      api_key=os.getenv("DEEPSEEK_API_KEY"),
                      base_url=os.getenv("DEEPSEEK_BASE_URL"))
search_tool=TavilySearch(
    max_results=3,
    topic="general"
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(BASE_DIR, "app", "db")
checkpoint_path = os.path.join(DB_DIR, "personal_chef.db")
store_path = os.path.join(DB_DIR, "store.db")
checkpoint_connect=sqlite3.connect(checkpoint_path,check_same_thread=False)
checkpoint=SqliteSaver(checkpoint_connect)
checkpoint.setup()

# SqliteStore 内部显式执行 BEGIN/COMMIT，连接必须使用 autocommit（isolation_level=None），
# 否则 setup() 里的 INSERT 会遗留隐式事务，导致下一次 BEGIN 报 "cannot start a transaction within a transaction"
store_connect=sqlite3.connect(store_path,check_same_thread=False,isolation_level=None)
store_connect.execute("PRAGMA journal_mode=WAL")
store_connect.execute("PRAGMA busy_timeout=5000")
store=SqliteStore(store_connect,
                  index=SqliteIndexConfig(
                      embed=get_embeddings(),
                      dims=1024
                  ))

SYSTEM_PROMPTS = {
"nutrition": """你是一名营养分析师，专注于分析食材/菜品的营养成分和热量。请按以下流程：
1. 识别食材/菜品：若有照片，识别所有可见食材并估算用量
2. 营养计算：优先调用调用 calculate_nutrition 计算营养成分,若calculate_nutrition工具返回结果不理想再可以调用web_search工具进行联网搜索
3. 营养评估：根据计算结果给出健康建议
⚠️ 避免幻觉：调用 calculate_nutrition 前，将食材换算为"1人份合理用量"例如：
- 鸡蛋 1.5~3 个、西红柿 1 个、肉类 80~180g、叶菜 80~200g、主食 80~150g
- 调料按整道菜适量，不要按人份倍增
请使用以下结构输出：
营养成分分析报告
- 各食材营养明细表格
- 营养汇总（总卡路里/蛋白质/脂肪/碳水/纤维）
- 营养建议（是否均衡、健康提示）
""",

"general": """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用similarity_search工具检索本地知识库的菜谱，web_search 工具联网搜索食谱，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。本地知识库中的食谱要被优先推荐。参考图片要给出连接，本地知识库的菜谱若没有连接可以输出无。
请严格按照流程，优先调用similarity_search工具检索本地知识库的菜谱，如果知识库检索出来的菜谱与用户提供的食材关联不大或菜谱数量不足再调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
特别提醒：如果用户的问题跟食谱没有啥关系或者关系不大你就是一个正常的聊天机器人，正常回答用户的问题就行了优先调用web_search工具进行回答，没有理想答案再自己发挥。
"""
}
tools=[similarity_search,web_search]
tools_by_name={t.name:t for t in tools}
LLM_with_tools=LLM.bind_tools(tools)
def run_agent(tools:list,System_prompt:str):
    inner_agent=create_agent(
        model=LLM,
        system_prompt=System_prompt,
        tools=tools,
        middleware=[SummarizationMiddleware(model=LLM, trigger=("messages", 20), keep=("messages", 6))],
        store=store,
    )
    return inner_agent

class UserPreference(BaseModel):
    context:str
def detect_intent(state:agentstate):
    messages=state.get("messages",[])
    if not messages:
        return {"intent":"general"}
    last_msg=messages[-1]
    if hasattr(last_msg, "content"):
        user_input = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    else:
        user_input = str(last_msg)
    nutrition_kw = ["卡路里", "热量", "营养", "多少卡", "蛋白质", "脂肪", "碳水",
                    "算一下", "计算", "分析营养", "热量多少", "营养成分", "健康吗", "减脂", "增肌"]
    for kw in nutrition_kw:
        if kw in user_input:
            logger.info(f"[detect_intent] → nutrition (匹配关键词: {kw})")
            return {"intent": "nutrition"}
    return {"intent": "general"}
manager=create_memory_store_manager(
    LLM,
    schemas=[UserPreference],
    namespace=("memories","{user_id}")
)
executer=ReflectionExecutor(manager,store=store)
general_llm = None
nutrition_llm = None
def general_llm_node(state:agentstate,config,writer:StreamWriter):
    global general_llm
    if general_llm is None:
        general_llm=run_agent(tools=[web_search,similarity_search],System_prompt=SYSTEM_PROMPTS["general"])
    user_id=config["configurable"]["user_id"]
    user_message=state["messages"][-1].content
    if not isinstance(user_message,str):
        user_message=str(user_message)
    memories=store.search(("memories",user_id),query=user_message,limit=5)
    memory_text="\n".join(
        m.value.get("context") for m in memories
        if isinstance(m.value,dict) and m.value.get("context")
    )
    messages=list(state["messages"])
    if memory_text:
        messages.append(HumanMessage(memory_text))
    if state.get("summary"):
        messages.append(SystemMessage("以下是历史信息的摘要:" + state["summary"]))
    # 内部 agent 是 LangGraph 图，输入必须是 state 字典
    # 流式转发其中的 token，同时用最后一个 values 事件拿到最终消息写回外层图
    final_state=None
    for mode,data in general_llm.stream(
        {"messages":messages},
        config={"recursion_limit":10},
        stream_mode=["messages","values"],
    ):
        if mode=="messages":
            chunk,_metadata=data
            if isinstance(chunk,AIMessageChunk) and chunk.content:
                writer(chunk.content)
        elif mode=="values":
            final_state=data
    response=final_state["messages"][-1]
    to_process={"messages":[state["messages"][-1]]+[response]}
    if to_process["messages"]:
        executer.submit(to_process,
                        after_seconds=0,
                        config={"configurable":{"user_id":user_id}})
    return {"messages": [response]}
def nutrition_node(state:agentstate,config,writer:StreamWriter):
    global nutrition_llm
    if nutrition_llm is None:
        nutrition_llm=run_agent(tools=[web_search,calculate_nutrition],System_prompt=SYSTEM_PROMPTS["nutrition"])
    user_id=config["configurable"]["user_id"]
    user_message=state["messages"][-1].content
    if not isinstance(user_message,str):
        user_message=str(user_message)
    memories=store.search(("memories",user_id),query=user_message,limit=5)
    memory_text="\n".join(
        m.value.get("context") for m in memories
        if isinstance(m.value,dict) and m.value.get("context")
    )
    messages=list(state["messages"])
    if memory_text:
        messages.append(HumanMessage(memory_text))
    if state.get("summary"):
        messages.append(SystemMessage("以下是历史信息的摘要:"+state["summary"]))
    final_state=None
    for mode,data in nutrition_llm.stream(
        {"messages":messages},
        config={"recursion_limit":10},
        stream_mode=["messages","values"],
    ):
        if mode=="messages":
            chunk,_metadata=data
            if isinstance(chunk,AIMessageChunk) and chunk.content:
                writer(chunk.content)
        elif mode=="values":
            final_state=data
    response=final_state["messages"][-1]
    to_process={"messages":[state["messages"][-1]]+[response]}
    if to_process["messages"]:
        executer.submit(to_process,
                        after_seconds=0,
                        config={"configurable": {"user_id": user_id}})
    return {"messages": [response]}
def summarize_node(state: agentstate):
    messages = state["messages"]
    keep_last_n = 4
    to_summarize = messages[:-keep_last_n] if len(messages) > keep_last_n else []
    if not to_summarize:
        return {}
    prev_summary=state.get("summary","") or ""
    summary_prompt = "请简要总结以下对话，保留关键信息和事实：\n\n"
    if prev_summary:
        summary_prompt += f"【已有历史摘要，请合并更新，不要丢失其中的事实】：\n{prev_summary}\n\n"
    summary_prompt += "【新增对话】：\n"
    for msg in to_summarize:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        summary_prompt += f"{role}: {msg.content}\n"
    summary_response = LLM.invoke([SystemMessage(content=summary_prompt)])
    # 用 RemoveMessage 删除所有被压缩的旧消息
    removals = [RemoveMessage(id=msg.id) for msg in to_summarize if msg.id]
    # 返回更新：新的摘要 + 要删除的消息列表
    return {
        "summary": summary_response.content,
        "messages": removals,  # add_messages reducer 会处理这个删除操作
    }
def summarize_route(state:agentstate):
    messages = state["messages"]
    if len(messages)>20:
        return "summarize_node"
    else:
        return END
builder=StateGraph(agentstate)
builder.add_node("general",general_llm_node)
builder.add_node("nutrition",nutrition_node)
builder.add_node("detect_intent",detect_intent)
builder.add_node("summarize_node",summarize_node)
builder.add_edge(START,"detect_intent")
builder.add_conditional_edges("detect_intent",lambda state:state["intent"],{"general": "general", "nutrition": "nutrition"})
builder.add_conditional_edges("general",summarize_route)
builder.add_conditional_edges("nutrition",summarize_route)
graph=builder.compile(checkpointer=checkpoint,store=store)
# 第一轮
async def search_recipes(prompt:str,image:str,thread_id:str,user_id:str):
    config = {"configurable":
                {"thread_id": thread_id,
                 "user_id":user_id}
              }
    logger.info(f"[用户]:{prompt}，“图片:{image}，thread_id:{thread_id}")
    if image and image.strip()!="":
        message=HumanMessage(content=[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":image}}
        ])
    else:
        message=HumanMessage(content=prompt)
    # SqliteSaver 不支持 astream，因此把同步 graph.stream 放到工作线程执行，
    # 再通过队列把 token 交回事件循环，避免阻塞整个 ASGI 事件循环
    queue:asyncio.Queue=asyncio.Queue()
    loop=asyncio.get_running_loop()
    end=object()

    def _produce():
        try:
            for chunk in graph.stream({"messages":[message]},config,stream_mode="custom"):
                loop.call_soon_threadsafe(queue.put_nowait,chunk)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait,e)
            logger.error(f"[流式对话失败] {e}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait,end)

    threading.Thread(target=_produce,daemon=True).start()
    while True:
        item=await queue.get()
        if item is end:
            break
        if isinstance(item,Exception):
            raise item
        yield item
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
"""def tool_node(state:messages_agentstate):
    message=state["messages"][-1]
    messages=[]
    for tool_call in message.tool_calls:
        _name=tool_call["name"]
        _args=tool_call["args"]
        _id=tool_call["id"]
        result=tools_by_name[_name].invoke(_args)
        messages.append(ToolMessage(content=result,tool_call_id=_id))
    return {"messages":messages}"""


"""def should_continue(state:messages_agentstate):
    last_message=state["messages"][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tool_call"
    else:
        return END"""