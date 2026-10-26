import httpx
from functools import lru_cache
from ragas.metrics.collections import Faithfulness,AnswerRelevancy,ContextPrecision,ContextEntityRecall,NoiseSensitivity,ContextRecall
from app.RAG.vector_store import get_vector_store,reset_vector,load_recipes
from app.RAG.rewrite import rag_search
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime,tool
import asyncio
import os
import time
import json
from ragas import Dataset
from app.common.logger import logger
from dotenv import load_dotenv
from langchain.agents import create_agent,AgentState
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI,OpenAI
load_dotenv()
llm=init_chat_model(model="deepseek-flash",
                      model_provider="openai",
                      api_key=os.getenv("DEEPSEEK_API_KEY"),
                      base_url=os.getenv("DEEPSEEK_BASE_URL"))
def vector_store_test():
    reset_vector()
    docs = load_recipes("E:/develop/Python/ai_chef/app/RAG/resource/菜谱.json")
    vector_stores=get_vector_store()
    vector_stores.add_documents(docs[1],ids=docs[0])
    text = rag_search("鸡肉")
    return text
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
context_store:dict[str,list[str]]={}
@tool
def search_knowledge_base(query:str,runtime:ToolRuntime[AgentState])->str:
    """搜索本地知识库，通过RAG向量检索的方式获取菜谱
    Args:
        query:用户提问的食材或关于食谱的问题
    Returns:
        知识库里的菜谱列表
    """
    docs =rag_search(query,3,3)
    sample_id = runtime.config.get("configurable", {}).get("sample_id")
    key=sample_id or runtime.state['messages'][0].content
    new_contents=[doc.page_content for doc in docs]
    existing=context_store.get(key,[])
    merged = list(dict.fromkeys(existing + new_contents))
    context_store[key] = merged
    logger.info(f"id={key} 召回 {len(docs)} 篇，累计上下文 {len(merged)} 条")
    return "\n\n".join(new_contents)
system_prompts="""
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1. 有关食谱的问题**只能**调用`search_knowledge_base`工具来查阅相关文档，根据文档信息回答问题
    2. **禁止**添加任何你自己的知识、理解或推断
    3. **禁止**扩展、举例或给出建议
    4. 如果参考文档中没有相关信息，直接回复"根据提供的资料，无法回答此问题"
    5. 回答要简洁，直接引用原文，不要改写或总结
    """
rag_agent=create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt=system_prompts,
)
def create_dataset():
    dataset=Dataset(name="rag_eval",backend="local/csv",root_dir="app/RAG/resource")
    if os.path.exists("app/RAG/resource/rag_eval.csv"):
        dataset.reload()
    if len(dataset)>0:
        return dataset
    else:
        request_delay=2
        with open("app/RAG/resource/ragas_user_inputs.json", encoding="utf-8") as f:
            questions = json.load(f)
        with open("app/RAG/resource/ragas_references.json", encoding="utf-8") as f:
            references = json.load(f)
        ref_map = {item["id"]: item["reference"] for item in references}
        for i in questions:
            query=i["user_input"]
            context_store.pop(i["id"], None)
            response = rag_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"sample_id": i["id"]}},
            )
            answer = response["messages"][-1].content
            contexts=context_store.get(i["id"],[])
            dataset.append({
                "user_input": i["user_input"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ref_map[i["id"]],
            })
            logger.info(f"添加数据成功... ✓ (contexts: {len(contexts)}条)")
            time.sleep(request_delay)
        dataset.save()
        return dataset
@lru_cache(maxsize=1)
def get_evaluator():
    llm_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")

    )
    evaluator_llm = llm_factory(
        model="deepseek-flash",
        client=llm_client,
        max_tokens=8192,
        timeout=httpx.Timeout(
            120.0,
            connect=15,
            read=90.0,
        ),
        max_retries=3,
    )
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # 占位符，本地 Ollama 不校验
    )
    evaluator_embeddings = embedding_factory(
        model="modelscope.cn/Qwen/Qwen3-Embedding-0.6B-GGUF:latest",
        client=ollama_client
    )
    return evaluator_llm,evaluator_embeddings
async def test_dataset():
    evaluator_llm,evaluator_embeddings=get_evaluator()
    dataset = create_dataset()
    row = dataset.__getitem__(3)
    user_input = row.get("user_input")
    response = row.get("response")
    retrieved_contexts = row.get("retrieved_contexts")
    reference = row.get("reference")
    print(f"========开始评估：input:{user_input}===========")
    # ---- 检索阶段指标 ----
    cp = ContextPrecision(llm=evaluator_llm)

    cp_result = await cp.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"运行完成context precision评估,score: {cp_result},进度1/6\n")

    cr = ContextRecall(llm=evaluator_llm)
    cr_result =await cr.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"运行完成context recall评估,score: {cr_result},进度2/6\n")

    cer = ContextEntityRecall(llm=evaluator_llm)
    cer_result =await cer.ascore(
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"运行完成ContextEntityRecall评估,score: {cer_result},进度3/6\n")
    # ---- 生成阶段指标 ----
    faith = Faithfulness(llm=evaluator_llm)

    faith_result =await faith.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"运行完成Faithfulness评估,score: {faith_result},进度4/6\n")

    ar = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ar_result = await ar.ascore(
        user_input=user_input,
        response=response,
    )
    print(f"运行完成AnswerRelevancy评估,score: {ar_result},进度5/6\n")

    ns = NoiseSensitivity(llm=evaluator_llm)
    ns_result =await ns.ascore(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"运行完成NoiseSensitivity评估,score: {ns_result},进度6/6\n")
    print(f"========================评估完成==========================")
    return 0
from ragas import experiment
from pydantic import BaseModel
class EvalResult(BaseModel):
    context_recall:float
    faithfulness:float
@experiment(experiment_model=EvalResult)
async def run_rag_eval(row):
    evaluator_llm, evaluator_embeddings = get_evaluator()
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = row.get("retrieved_contexts")
    response = row.get("response")
    cr = ContextRecall(llm=evaluator_llm).ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    faith = Faithfulness(llm=evaluator_llm).ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,)
    cr_result, faith_result=await asyncio.gather(cr,faith)
    print(f"完成评估")
    # 封装所有结果
    return EvalResult(
        context_recall=cr_result.value,
        faithfulness=faith_result.value
    )
if __name__ == "__main__":
    results = asyncio.run(run_rag_eval.arun(create_dataset(), name="rag_baseline"))