# 🍳 AI 私厨（Personal Chef）

一个基于 **FastAPI** + **LangGraph** 的个人 AI 大厨助手。用户上传食材图片或输入文字描述，AI 大厨会解析食材、联网搜索菜谱，并推荐多道菜品及制作流程。

> 学习向项目：用于探索 LangChain / LangGraph Agent 的开发与部署流程。

---

## ✨ 核心功能

- **文字 / 图片识别食材**：基于 DeepSeek 视觉模型解析用户图片中的食材，分析色泽、新鲜度并列出可用食材清单。
- **联网搜菜谱**：自动调用 Tavily 搜索联网查询菜谱与参考图，搜索失败时由模型自行发挥。
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
| 大模型 | DeepSeek（V4 Flash Vision，OpenAI 兼容接口） |
| 联网搜索 | Tavily |
| 会话记忆 | SqliteSaver（SQLite checkpoint，多线程安全） |
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
│   │   └── personal_chef.py  # AI 大厨 Agent：模型初始化、工具、记忆、流式生成
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py       # 对话接口（流式 / 历史查询 / 清空）
│   │       └── oss.py        # OSS 预签名上传接口
│   ├── common/
│   │   └── logger.py         # 日志配置
│   ├── models/
│   │   └── schemas.py        # Pydantic 请求模型
│   ├── db/                   # SQLite 会话数据库（运行时生成）
│   └── static/               # 前端静态资源（Next.js 构建产物）
├── .env.example              # 环境变量示例（见下方配置）
├── pyproject.toml            # 项目依赖（uv）
└── uv.lock                   # 依赖锁文件
```

---

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（包管理与虚拟环境）

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入真实密钥：

```env
# DeepSeek 大模型
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Tavily 联网搜索
TAVILY_API_KEY=your_tavily_key

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
```

### 4. 启动服务

```bash
python -m app.main
# 或者指定 uv：
uv run python -m app.main
```

默认监听 `http://127.0.0.1:8001`（支持热重载）。打开浏览器即可使用，或通过 `/docs`（Swagger）查看接口文档。

---

## 🔌 API 接口

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
- 通过 `TavilySearch` 封装 `web_search` 工具，供 Agent 联网检索菜谱。
- 使用 `SqliteSaver`（连接 `db/personal_chef.db`）持久化多轮对话，支持按 `thread_id` 恢复与清空会话。
- `create_agent` 结合系统提示词（大厨角色、目标、能力要求与限制）构建 Agent。
- `.stream(..., stream_mode="messages")` 按 token 流式产出回复。

---

## 🛠️ 常见问题

- **首次运行无 `db/` 目录**：代码会在 Agent 初始化时自动创建 `db/personal_chef.db`，请确保运行目录具备写入权限。
- **中文乱码**：Windows 下请确保设置 `PYTHONUTF8=1`。
- **前端空白**：`app/static` 存放的是前端构建产物，若目录为空则服务仅返回后端提示文案。

---

## 📄 许可证

本项目仅供学习研究使用。