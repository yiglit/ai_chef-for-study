# 🍳 AI 私厨（Personal Chef）

一个基于 **FastAPI** + **LangGraph** 的个人 AI 大厨助手。用户上传食材图片或输入文字描述，AI 大厨会解析食材、检索本地菜谱知识库（RAG）、必要时联网搜索，并推荐多道菜品及制作流程。

> 学习向项目：用于探索 LangChain / LangGraph Agent、Agentic RAG 的开发与部署流程。

---

## ✨ 核心功能

- **文字 / 图片识别食材**：基于 DeepSeek 视觉模型解析用户图片中的食材，分析色泽、新鲜度并列出可用食材清单。
- **本地菜谱知识库检索（RAG）**：MinerU 解析 PDF → 递归分块 → 向量化入库 Milvus，检索前用 HyDE 改写 Query，采用稠密向量 + BM25 混合检索并以 RRF 融合排序。
- **联网搜菜谱**：本地知识库命中不足或无结果时，自动调用 Tavily 联网查询菜谱与参考图，最后才由模型自行发挥。
- **智能推荐**：根据制作难度与营养价值打分排序，数值高者优先；输出结构化数据（食谱信息、评分、推荐理由、参考图）。
- **多轮记忆**：基于 `langgraph-checkpoint-sqlite` 保存会话上下文，支持按 `thread_id` 管理对话历史。
- **流式对话**：前端通过 SSE（`text/event-stream`）流式接收回复。
- **图片上传**：对接阿里云 OSS，生成预签名上传 URL，方便前端直传图片。

---

## 🧱 技术栈

| 分类 | 技术 |
| --- | --- |
| 后端 | FastAPI · Uvicorn · pydantic v2 |
| AI Agent | LangChain · LangGraph · langgraph-cli |
| 大模型 | DeepSeek（Flash Vision，OpenAI 兼容接口） |
| 联网搜索 | Tavily |
| RAG 文档解析 | MinerU（PDF 版面解析） |
| 向量库 | Milvus / Zilliz Cloud（稠密 + 稀疏 BM25 混合检索） |
| Embedding | Ollama · Qwen3-Embedding-0.6B（1024 维） |
| 会话记忆 | SqliteSaver（SQLite checkpoint，多线程安全） |
| 工具集成 | langchain-mcp-adapters（MCP 客户端） |
| 对象存储 | 阿里云 OSS v2 SDK（预签名直传） |
| 可观测 | LangSmith Tracing · Prometheus Client |
| 依赖管理 | uv（清华 PyPI 源） |
| 运行环境 | Python >= 3.11 |

---

## 📁 项目结构

```
ai_chef/
├── app/
│   ├── main.py               # FastAPI 入口：CORS、路由挂载、静态资源与 SPA fallback
│   ├── agents/
│   │   ├── personal_chef.py  # AI 大厨 Agent：模型初始化、工具、记忆、流式生成
│   │   └── Nutrition_ans.py  # 学习用示例：MCP 客户端 Demo（未接入主流程）
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py       # 对话接口（流式 / 历史查询 / 清空）
│   │       └── oss.py        # OSS 预签名上传接口
│   ├── RAG/
│   │   ├── Mineru.py         # 用 MinerU 解析 PDF → 分块 → 入库（独立脚本）
│   │   ├── vector_store.py   # Milvus 向量库封装（混合检索）、菜谱 JSON 导入
│   │   ├── embeddings.py     # Ollama Embedding（Qwen3-Embedding-0.6B）
│   │   ├── HyDE.py           # Query 改写 + 向量召回（rag_search 入口）
│   │   ├── search.py         # 本地检索调试脚本
│   │   └── resource/         # 菜谱 JSON、PDF 语料
│   ├── common/
│   │   └── logger.py         # 日志配置
│   ├── models/
│   │   └── schemas.py        # Pydantic 请求模型
│   ├── db/                   # SQLite 会话数据库（运行时生成）
│   └── static/               # 前端静态资源（Next.js 构建产物）
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
- 一个 Milvus / Zilliz 实例（用于存放菜谱向量）

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件并填入真实密钥（该文件已被 `.gitignore` 忽略）：

```env
# DeepSeek 大模型
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Tavily 联网搜索
TAVILY_API_KEY=your_tavily_key

# MinerU 文档解析
MINERU_TOKEN=your_mineru_token

# Milvus / Zilliz Cloud（向量检索）
ZILI_URI=your_zilliz_uri
ZILI_TOKEN=your_zilliz_token

# LangSmith 可观测（可选）
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=cooker_agent

# 强制使用 UTF-8（Windows 下处理中文）
PYTHONUTF8=1

# 阿里云 OSS（图片上传）
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

## � RAG 知识库构建

主流程依赖本地菜谱知识库，首次使用需先入库：

1. **准备语料**：将 PDF 放入 `app/RAG/resource/`，或使用已有的 `菜谱.json` / `食谱.json`。
2. **启动 Ollama 并拉取 Embedding 模型**：`ollama pull modelscope.cn/Qwen/Qwen3-Embedding-0.6B-GGUF:latest`。
3. **执行入库脚本**（以 PDF 为例，需在 `app/RAG` 目录下运行）：

```bash
cd app/RAG
python Mineru.py
```

- `Mineru.py`：MinerU 解析 PDF → `RecursiveCharacterTextSplitter`（chunk 500 / overlap 100）→ 写入 Milvus。
- `vector_store.py`：提供 `load_recipes()` 与 `reset_vector()`，可将 `菜谱.json` 结构化菜谱批量导入或重建 collection。
- `search.py`：入库后调试检索效果的独立脚本。

> 检索链路：`rag_search()` → `rewrite_query()`（HyDE 式 Query 改写）→ `vector_call()`（Milvus 稠密 + BM25 稀疏，`ranker_type="rrf"` 融合）。

---

## �🔌 API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/chat/stream` | 流式对话（SSE）。请求体含 `message`、`image_url`（可选）、`thread_id` |
| GET | `/api/v1/chat/messages?thread_id=xxx` | 获取指定会话的历史消息 |
| DELETE | `/api/v1/chat/messages?thread_id=xxx` | 清空指定会话的历史消息 |
| GET | `/api/v1/oss/presign?filename=xxx` | 获取 OSS 预签名上传 URL 与访问 URL |

### 流式对话示例

```bash
curl -N -X POST http://127.0.0.1:8001/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "冰箱里有鸡蛋、番茄和土豆",
    "image_url": "",
    "thread_id": "demo-thread"
  }'
```

---

## 🧠 核心逻辑说明

`app/agents/personal_chef.py` 中定义了 AI 大厨 Agent：

- 使用 `init_chat_model` 创建 DeepSeek 视觉多模态模型（OpenAI 兼容接口）。
- 注册两个工具：`similarity_search`（封装 `app/RAG/HyDE.rag_search` 检索本地菜谱）与 `web_search`（封装 `TavilySearch`）。
- 通过系统提示词约定策略：**优先检索本地知识库 → 命中不足再联网 → 最后才自行发挥**，并按营养价值与制作难度打分排序、输出结构化报告。
- 使用 `SqliteSaver`（连接 `db/personal_chef.db`）持久化多轮对话，支持按 `thread_id` 恢复与清空会话。
- `create_agent` 构建 Agent 后，`.stream(..., stream_mode="messages")` 按 token 流式产出回复。
- 通过 `MultiServerMCPClient`（`langchain-mcp-adapters`）预留 MCP 远程工具接入能力。

---

## 🛠️ 常见问题

- **会话数据库位置**：`SqliteSaver` 连接的路径为 `db/personal_chef.db`（相对于**启动命令时的工作目录**），仓库中已内置 `app/db/`，请确保启动目录与数据库路径一致且具备写入权限。
- **中文乱码**：Windows 下请确保设置 `PYTHONUTF8=1`。
- **检索无结果**：确认 Ollama 已启动、Embedding 模型已拉取，且 Milvus collection 已完成入库（见上方「RAG 知识库构建」）。
- **前端空白**：`app/static` 存放的是前端构建产物，若目录为空则服务仅返回后端提示文案。
- **`pyproject.toml` 中的 `jq`**：为早期实验遗留依赖，若 `uv sync` 报版本不存在可直接移除。

---

## 📄 许可证

本项目仅供学习研究使用。
