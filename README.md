# 🍳 AI 私厨（Personal Chef）

一个基于 **FastAPI** + **LangChain / LangGraph** 的个人 AI 大厨助手。用户上传食材图片或输入文字描述，AI 大厨会解析食材、检索本地菜谱知识库（RAG，含重排序）、必要时联网搜索，并推荐多道菜品及制作流程；同时支持营养计算与长时记忆。

> 学习向项目：用于探索 LangChain Agent、Agentic RAG、LangGraph 多节点编排与多轮/长时记忆的开发与部署流程。

---

## ✨ 核心功能

- **文字 / 图片识别食材**：基于 DeepSeek 多模态模型解析用户图片中的食材，分析色泽、新鲜度并列出可用食材清单。
- **意图路由**：入口节点 `detect_intent` 按关键词将请求分流到「菜谱推荐」或「营养分析」两条子链路。
- **本地菜谱知识库检索（RAG）**：菜谱结构化文本 → 向量化入库 Milvus；检索前用 LLM 把口语化问题改写为检索关键词，Milvus 稠密向量 + BM25 稀疏混合检索（RRF 融合），再用 CrossEncoder（`bge-reranker-v2-m3`）精排。
- **联网搜菜谱**：本地知识库命中不足或无结果时，自动调用 Tavily 联网查询菜谱与参考图，最后才由模型自行发挥。
- **智能推荐**：根据制作难度与营养价值打分排序，数值高者优先；输出结构化数据（食谱信息、评分、推荐理由、参考图）。
- **营养计算**：`calculate_nutrition` 工具按食材与份数计算卡路里、蛋白质、脂肪、碳水与纤维，并给出营养建议。
- **文档知识库（MinerU）**：把 PDF 语料用 MinerU 解析为 Markdown，按 Markdown 结构切分后写入独立的 `mineru_docs` 向量集合。
- **多轮记忆**：基于 `langgraph-checkpoint-sqlite` 保存会话上下文，支持按 `thread_id` 管理对话历史。
- **长时记忆**：通过 `langmem` + `SqliteStore` 抽取并检索用户偏好（`UserPreference`），跨会话注入上下文。
- **上下文摘要压缩**：对话过长时自动摘要压缩，保留近期关键内容（`personal_chef.py` 用 `SummarizationMiddleware`，`graph_chef.py` 用自建 `summarize_node`）。
- **流式对话**：前端通过 SSE（`text/event-stream`）流式接收回复。
- **图片上传**：对接阿里云 OSS，生成预签名上传 URL，方便前端直传图片。

---

## 🧱 技术栈

| 分类 | 技术 |
| --- | --- |
| 后端 | FastAPI · Uvicorn · pydantic v2 |
| AI Agent | LangChain（`create_agent`）· LangGraph（`StateGraph` 多节点编排） |
| 大模型 | DeepSeek `deepseek-flash`（OpenAI 兼容接口，由 `init_chat_model` 创建） |
| 联网搜索 | Tavily（`langchain-tavily`） |
| 向量库 | Milvus / Zilliz Cloud（稠密 + 稀疏 BM25 混合检索，RRF 融合） |
| Embedding | Ollama · Qwen3-Embedding-0.6B（1024 维） |
| 重排序 | sentence-transformers CrossEncoder（`BAAI/bge-reranker-v2-m3`） |
| 文本切分 | `langchain-text-splitters`（RecursiveCharacter / RecursiveJson） |
| 文档解析 | MinerU（`langchain-mineru`，PDF → Markdown） |
| 会话记忆 | SqliteSaver（SQLite checkpoint，多线程安全） |
| 长时记忆 | langmem + SqliteStore（语义记忆检索） |
| 上下文管理 | SummarizationMiddleware / 自建摘要节点 |
| 对象存储 | 阿里云 OSS v2 SDK（预签名直传） |
| 评测 | ragas（RAG 评估数据集与基线） |
| 可观测 | LangSmith Tracing（通过环境变量开启） |
| 依赖管理 | uv（清华 PyPI 源） |
| 运行环境 | Python >= 3.11 |

---

## 📁 项目结构

```
ai_chef/
├── app/
│   ├── main.py               # FastAPI 入口：CORS、路由挂载、静态资源与 SPA fallback
│   ├── agents/
│   │   ├── graph_chef.py     # 【主服务】LangGraph 图 Agent：意图路由 + 菜谱/营养双链路 + 长时记忆 + 流式输出
│   │   └── personal_chef.py  # 早期单 Agent 版本（create_agent + SummarizationMiddleware）
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py       # 对话接口（流式 / 历史查询 / 清空）
│   │       └── oss.py        # OSS 预签名上传接口
│   ├── RAG/
│   │   ├── rewrite.py        # 检索入口：Query 改写 + Milvus 混合召回 + CrossEncoder 精排（rag_search）
│   │   ├── vector_store.py   # Milvus 向量库封装（稠密 + BM25 混合检索）、菜谱 JSON 装载与重置
│   │   ├── embeddings.py     # Ollama Embedding（Qwen3-Embedding-0.6B，1024 维）
│   │   ├── Mineru.py         # MinerU 文档入库流水线：PDF → Markdown → 切分 → 写入 mineru_docs
│   │   └── resource/         # 菜谱 JSON、PDF 语料、评测数据集与解析输出
│   ├── common/
│   │   └── logger.py         # 日志配置
│   ├── models/
│   │   └── schemas.py        # Pydantic 请求模型
│   ├── db/                   # SQLite 会话库（personal_chef.db）与记忆库（store.db），运行时生成
│   └── static/               # 前端静态资源（Next.js 构建产物）
├── docs/                     # 项目文档（项目全解、面试手册等）
├── .env                      # 环境变量（需自行创建，已被 .gitignore 忽略）
├── pyproject.toml            # 项目依赖（uv）
└── uv.lock                   # 依赖锁文件
```

---

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（包管理与虚拟环境）
- [Ollama](https://ollama.com/)（本地 Embedding 推理，RAG 检索需要）
- 一个 Milvus / Zilliz 实例（用于存放菜谱与文档向量）
- MinerU 服务或凭证（仅在构建 PDF 知识库时需要）

### 2. 安装依赖

```bash
uv sync
```

> 依赖已在 `pyproject.toml` 中声明（含 `langchain-ollama`、`langchain-milvus`、`langchain-text-splitters`、`langchain-mineru`、`langmem`、`sentence-transformers` 等），`uv sync` 后即可直接运行。

### 3. 配置环境变量

在项目根目录创建 `.env` 文件并填入真实密钥（该文件已被 `.gitignore` 忽略）：

```env
# DeepSeek 大模型
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Tavily 联网搜索
TAVILY_API_KEY=your_tavily_key

# Milvus / Zilliz Cloud（向量检索）
ZILI_URI=your_zilliz_uri
ZILI_TOKEN=your_zilliz_token

# LangSmith 可观测（可选）
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=cooker_agent

# 强制使用 UTF-8（Windows 下处理中文）
PYTHONUTF8=1

# 阿里云 OSS（图片上传，依赖 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET 环境变量）
OSS_ACCESS_KEY_ID=your_oss_ak
OSS_ACCESS_KEY_SECRET=your_oss_sk
OSS_BUCKET=your_bucket_name
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
```

### 4. 启动服务

```bash
python -m app.main
# 或者指定 uv：
uv run python -m app.main
```

默认监听 `http://127.0.0.1:8001`（支持热重载）。打开浏览器即可使用，或通过 `/docs`（Swagger）查看接口文档。

---

## 📚 RAG 知识库构建

### 1. 菜谱知识库（结构化 JSON）

1. **准备语料**：使用 `app/RAG/resource/菜谱.json`（结构化菜谱：菜名 / 菜系 / 难度 / 烹饪时间 / 食材 / 标签 / 步骤 / 简介）。
2. **启动 Ollama 并拉取 Embedding 模型**：`ollama pull modelscope.cn/Qwen/Qwen3-Embedding-0.6B-GGUF:latest`。
3. **执行入库**：调用 [vector_store.py](file:///e:/develop/Python/ai_chef/app/RAG/vector_store.py) 中的 `load_recipes(file_path)` 读取并拼装文档，再用 `get_vector_store()` 写入 Milvus：

```python
from app.RAG.vector_store import load_recipes, get_vector_store

ids, docs = load_recipes("./app/RAG/resource/菜谱.json")
get_vector_store().add_documents(docs, ids=ids)
```

### 2. PDF 文档知识库（MinerU）

把 PDF 语料解析为 Markdown 后写入独立集合 `mineru_docs`（与菜谱集合互不影响），入口在 [Mineru.py](file:///e:/develop/Python/ai_chef/app/RAG/Mineru.py)：

```bash
# 默认解析 app/RAG/resource/浙江风味2.pdf，也可传入自定义 PDF 路径
python -m app.RAG.Mineru ./app/RAG/resource/浙江风味2.pdf
```

- 流程：`load_documents()`（MinerU 解析 PDF）→ 保存 Markdown 到 `resource/output/` → `split_documents()`（按 Markdown 结构切分、统一元数据）→ `upsert` 写入 Milvus（重复入库自动覆盖）。

### 3. 检索链路

- [rewrite.py](file:///e:/develop/Python/ai_chef/app/RAG/rewrite.py)：检索入口，`rag_search()` = `rewrite_query()`（LLM 将口语问题改写为 5~10 个检索关键词）→ `vector_call()`（Milvus 稠密 + BM25 稀疏，`ranker_type="rrf"` 融合）→ `re_rank()`（CrossEncoder `BAAI/bge-reranker-v2-m3` 精排后取 Top-N）。
- [vector_store.py](file:///e:/develop/Python/ai_chef/app/RAG/vector_store.py)：提供 `get_vector_store()`（Milvus + 中文 BM25 混合检索）、`load_recipes()`（菜谱 JSON → 文档）与 `reset_vector()`（重建 collection）。
- [embeddings.py](file:///e:/develop/Python/ai_chef/app/RAG/embeddings.py)：Ollama Embedding（Qwen3-Embedding-0.6B，1024 维）。

> 菜谱默认 collection 名为 `langchain_collection`，MinerU 文档集合为 `mineru_docs`。

---

## 🔌 API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/chat/stream` | 流式对话（SSE）。请求体含 `message`、`image_url`（可选）、`thread_id`、`user_id`（可选，默认 `default`） |
| GET | `/api/v1/chat/messages?thread_id=xxx` | 获取指定会话的历史消息 |
| DELETE | `/api/v1/chat/messages?thread_id=xxx` | 清空指定会话的历史消息 |
| GET | `/api/v1/oss/presign?filename=xxx` | 获取 OSS 预签名上传 URL、Content-Type 与访问 URL |

### 流式对话示例

```bash
curl -N -X POST http://127.0.0.1:8001/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "冰箱里有鸡蛋、番茄和土豆",
    "image_url": "",
    "thread_id": "demo-thread",
    "user_id": "user-001"
  }'
```

---

## 🧠 核心逻辑说明

### 主服务图 Agent（[graph_chef.py](file:///e:/develop/Python/ai_chef/app/agents/graph_chef.py)）

`chat.py` 实际调用的是 `graph_chef.py`，它是一个 LangGraph `StateGraph`：

- **状态**：`agentstate`（`messages` / `summary` / `intent` / `thread_id`），消息使用 `add_messages` reducer。
- **入口路由**：`START → detect_intent`，按关键词（热量、营养、蛋白质、计算…）判定 `intent`，条件边分流到 `general`（菜谱推荐）或 `nutrition`（营养分析）。
- **双链路**：`general_llm_node` 与 `nutrition_node` 各自懒加载一个 `create_agent` 子 Agent，分别绑定 `similarity_search` / `web_search` 与 `calculate_nutrition` / `web_search`。
- **长时记忆**：`langmem.create_memory_store_manager` 抽取 `UserPreference`，`store.search(("memories", user_id))` 在生成前注入相关记忆，`ReflectionExecutor` 在回复后异步写入。
- **摘要压缩**：`summarize_node` 在消息超过 20 条时总结旧消息并用 `RemoveMessage` 删除，避免上下文无限增长。
- **流式输出**：节点内用 `StreamWriter` 透传 token，外层 `search_recipes()` 把同步 `graph.stream(..., stream_mode="custom")` 放到工作线程并通过队列交回事件循环（`SqliteSaver` 不支持 `astream`）。
- **持久化**：`SqliteSaver`（`app/db/personal_chef.db`）保存会话，`SqliteStore`（`app/db/store.db`）保存长时记忆；`getmessages()` 读历史，`clear_messages()` 通过 `checkpoint.delete_thread()` 清空会话。

### 早期单 Agent 版本（[personal_chef.py](file:///e:/develop/Python/ai_chef/app/agents/personal_chef.py)）

- 用 `init_chat_model(model="deepseek-flash", model_provider="openai", ...)` 创建 DeepSeek 多模态模型。
- `create_agent` + `SqliteSaver` + `SummarizationMiddleware`（`trigger=("fraction", 0.8)` / `keep=("fraction", 0.2)`）搭建单轮 Agent，`.stream(..., stream_mode="messages")` 按 token 流式产出。

---

## 🛠️ 常见问题

- **会话数据库位置**：`graph_chef.py` 通过项目根目录定位，数据库固定在 `app/db/personal_chef.db` 与 `app/db/store.db`；`personal_chef.py` 使用相对路径 `db/personal_chef.db`（相对于启动命令时的工作目录），请确保启动目录与数据库路径一致且具备写入权限。
- **SQLite 并发写入报错**：`store.db` 已开启 WAL 与 `busy_timeout`，checkpoint 连接使用 `check_same_thread=False`；如仍出现 `database is locked`，请确认没有多个进程同时写入同一文件。
- **中文乱码**：Windows 下请确保设置 `PYTHONUTF8=1`。
- **检索无结果**：确认 Ollama 已启动、Embedding 模型已拉取，且 Milvus collection 已完成入库（见上方「RAG 知识库构建」）。
- **首次重排序较慢**：`re_rank` 首次调用会下载并加载 `BAAI/bge-reranker-v2-m3` 模型。
- **前端空白**：`app/static` 存放的是前端构建产物，若目录为空则服务仅返回后端提示文案。
- **OSS 上传报错**：`oss.py` 中 Region 固定为 `cn-hangzhou`，且凭证通过环境变量提供（`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`），请确认配置一致。

---

## 📄 许可证

本项目仅供学习研究使用。
