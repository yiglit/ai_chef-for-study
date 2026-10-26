# test_agent.py
"""针对 app/agents/graph_chef.py 的手工测试脚本。

用法：
    python test_agent.py            # 只运行离线测试（不依赖外部服务）
    python test_agent.py --with-llm # 追加运行集成测试（需要 LLM / Ollama / Milvus / Tavily）

说明：
    - 离线测试覆盖营养计算、意图识别、图结构、会话读写等纯本地逻辑。
    - 集成测试会真实调用大模型和外部工具，会产生网络请求与耗时。
"""
import asyncio
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph_chef import (
    SYSTEM_PROMPTS,
    calculate_nutrition,
    clear_messages,
    detect_intent,
    getmessages,
    graph,
    run_agent,
    search_recipes,
    similarity_search,
    web_search,
)


def _new_thread_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def _state(prompt: str, intent: str = "general", thread_id: str = "test") -> dict:
    return {
        "messages": [HumanMessage(prompt)],
        "summary": "",
        "intent": intent,
        "thread_id": thread_id,
    }


# ==================== 离线测试：营养计算工具 ====================

def test_calculate_nutrition_basic():
    """多食材 + 多人份：应输出明细表格、汇总与建议。"""
    result = calculate_nutrition.invoke(
        {"ingredients": "鸡胸肉200g, 西兰花150g", "servings": 2}
    )
    assert "营养成分分析报告（2人份）" in result
    assert "| 鸡胸肉 | 200g |" in result
    assert "| 西兰花 | 150g |" in result
    # 165*2*2 + 34*1.5*2 = 660 + 102 = 762
    assert "762.0" in result
    assert "总卡路里" in result


def test_calculate_nutrition_default_weight():
    """未标注重量时应按 100g 计算。"""
    result = calculate_nutrition.invoke({"ingredients": "鸡蛋", "servings": 1})
    assert "| 鸡蛋 | 100g |" in result
    assert "155.0" in result  # 每 100g 鸡蛋的卡路里


def test_calculate_nutrition_gram_unit():
    """支持“克”单位解析。"""
    result = calculate_nutrition.invoke({"ingredients": "大米150克", "servings": 1})
    assert "| 大米 | 150g |" in result
    assert "547.5" in result  # 365 * 1.5


def test_calculate_nutrition_unknown_ingredient():
    """库中不存在的食材应给出提示而不是报错。"""
    result = calculate_nutrition.invoke({"ingredients": "榴莲300g", "servings": 1})
    assert "未找到该食材的营养数据" in result


def test_calculate_nutrition_exact_match_priority():
    """精确匹配优先：三文鱼不应被“鱼”抢先匹配。

    注意：当前 graph_chef.py 使用字典顺序的子串匹配，
    该用例会失败，可用于暴露食材匹配优先级问题。
    """
    result = calculate_nutrition.invoke({"ingredients": "三文鱼100g", "servings": 1})
    assert "208.0" in result, "三文鱼应按 208 kcal/100g 计算，而不是鱼的 165 kcal"


# ==================== 离线测试：意图识别 ====================

def test_detect_intent_nutrition():
    """命中营养关键词时应路由到 nutrition 节点。"""
    for prompt in ["这顿饭有多少卡路里", "帮我算一下蛋白质", "减脂期吃这个健康吗"]:
        intent = detect_intent(_state(prompt))["intent"]
        assert intent == "nutrition", f"“{prompt}”应识别为 nutrition，实际为 {intent}"


def test_detect_intent_general():
    """普通菜谱请求应路由到 general 节点。"""
    assert detect_intent(_state("推荐一道简单好做的菜"))["intent"] == "general"
    assert detect_intent(_state("冰箱里有鸡胸肉和西兰花"))["intent"] == "general"


def test_detect_intent_empty_messages():
    """空消息列表应默认走 general。"""
    assert detect_intent({"messages": []})["intent"] == "general"


# ==================== 离线测试：图结构与 Agent 构建 ====================

def test_graph_structure():
    """图中应包含意图识别与两个业务节点。"""
    node_names = set(graph.get_graph().nodes.keys())
    assert {"detect_intent", "general", "nutrition"} <= node_names


def test_run_agent_builds():
    """run_agent 应能正常构建内部 Agent（不触发网络调用）。"""
    agent = run_agent(tools=[calculate_nutrition], System_prompt=SYSTEM_PROMPTS["nutrition"])
    assert agent is not None


# ==================== 离线测试：会话读写 ====================

def test_getmessages_empty_thread():
    """不存在的会话应返回空列表。"""
    assert getmessages(_new_thread_id()) == []


def test_clear_messages_no_error():
    """清空不存在的会话不应抛异常。"""
    thread_id = _new_thread_id()
    clear_messages(thread_id)
    assert getmessages(thread_id) == []


# ==================== 集成测试：真实对话（需要外部依赖） ====================

async def _collect_reply(prompt: str, thread_id: str, user_id: str, image: str = "") -> str:
    """把 search_recipes 的流式输出拼成完整文本。"""
    chunks = []
    async for chunk in search_recipes(prompt, image, thread_id, user_id):
        chunks.append(chunk)
    return "".join(chunks)


async def test_nutrition_chat_flow():
    """营养意图端到端：流式有输出 + 历史可读 + 可清空。"""
    thread_id = _new_thread_id()
    reply = await _collect_reply(
        "鸡蛋2个、番茄1个、大米100g，帮我算算热量", thread_id, "test-user"
    )
    assert reply.strip(), "营养意图没有流式输出"
    assert any(kw in reply for kw in ("营养", "卡路里", "热量", "蛋白质")), reply

    history = getmessages(thread_id)
    assert any(item["role"] == "user" for item in history)
    assert any(item["role"] == "assistant" for item in history)

    clear_messages(thread_id)
    assert getmessages(thread_id) == []


async def test_general_chat_flow_and_multi_turn():
    """普通菜谱意图端到端：两轮对话 + 历史累计。"""
    thread_id = _new_thread_id()
    user_id = "test-user"
    first = await _collect_reply(
        "冰箱里有鸡胸肉和西兰花，推荐一道简单的菜", thread_id, user_id
    )
    assert first.strip(), "普通意图没有流式输出"

    second = await _collect_reply("那这道菜大概要多久能做完？", thread_id, user_id)
    assert second.strip(), "第二轮没有流式输出"

    history = getmessages(thread_id)
    assert sum(1 for item in history if item["role"] == "user") >= 2

    clear_messages(thread_id)


async def test_similarity_search_tool():
    """本地知识库检索工具（依赖 Milvus + Ollama + 重排模型）。"""
    result = similarity_search.invoke({"query": "鸡肉 家常菜", "top_n": 2})
    assert isinstance(result, str) and result.strip()


async def test_web_search_tool():
    """联网搜索工具（依赖 TAVILY_API_KEY）。"""
    result = web_search.invoke({"query": "西红柿炒鸡蛋的做法"})
    assert result


# ==================== 运行入口 ====================

OFFLINE_TESTS = [
    test_calculate_nutrition_basic,
    test_calculate_nutrition_default_weight,
    test_calculate_nutrition_gram_unit,
    test_calculate_nutrition_unknown_ingredient,
    test_calculate_nutrition_exact_match_priority,
    test_detect_intent_nutrition,
    test_detect_intent_general,
    test_detect_intent_empty_messages,
    test_graph_structure,
    test_run_agent_builds,
    test_getmessages_empty_thread,
    test_clear_messages_no_error,
]

LLM_TESTS = [
    test_nutrition_chat_flow,
    test_general_chat_flow_and_multi_turn,
    test_similarity_search_tool,
    test_web_search_tool,
]


def _run_sync(name: str, fn) -> bool:
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def _run_async(name: str, coro_fn) -> bool:
    try:
        asyncio.run(coro_fn())
        print(f"[PASS] {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    with_llm = "--with-llm" in sys.argv
    passed = 0
    failed = 0

    print("========== 离线测试 ==========")
    for test in OFFLINE_TESTS:
        if _run_sync(test.__name__, test):
            passed += 1
        else:
            failed += 1

    if with_llm:
        print("\n========== 集成测试（真实调用 LLM / 外部工具） ==========")
        for test in LLM_TESTS:
            if _run_async(test.__name__, test):
                passed += 1
            else:
                failed += 1
    else:
        print("\n提示：使用 `python test_agent.py --with-llm` 可运行集成测试")

    print(f"\n结果：通过 {passed}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
