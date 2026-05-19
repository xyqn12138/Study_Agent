# Study Agent

基于 LangGraph 的个人超级知识库 Agent，通过多阶段 RAG 管线实现对教材、课件等学术文档的深度问答。提供 Web 界面，支持文件上传入库和流式对话。

## 架构概览

```
PDF / Markdown 文档
        │
        ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
 │  MinerU API  │────▶│  md_loader   │────▶│   md_splitter    │
 │  (PDF → MD)  │     │  (结构解析)   │     │  (四层分块)       │
 └──────────────┘     └──────────────┘     └────────┬─────────┘
                                                     │
                                                     ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │   Retriever  │◀────│  Reranker    │◀────│  data_embedding  │
  │  (混合检索)   │     │  (qwen3)     │     │  (向量化→Milvus)  │
  └──────┬───────┘     └──────────────┘     └──────────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────────────────────────────┐
  │ LangGraph    │     │ tools: search / context / image / web│
  │ ReAct Agent  │────▶│                                      │
  └──────┬───────┘     └──────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  FastAPI     │────▶│  Web 前端界面     │     │  Security Guard  │
  │  SSE 流式    │     │  (ChatGPT 风格)   │     │  (输入安全审计)   │
  └──────────────┘     └──────────────────┘     └──────────────────┘
```

## 四层分块体系

| 层级 | 角色 | 内容 | 用途 |
|------|------|------|------|
| **L1** 章 | 索引节点 | 仅标题 | 层级导航、溯源 |
| **L2** 小节 | 索引节点 | 仅标题 | 双阶段检索锚点 |
| **L3** 知识块 | 主召回层 | 语义段落（~1500字符） | 混合检索主要目标 |
| **L4** 证据块 | 细粒度层 | 短段落（~400字符） | 精确匹配、关键词命中 |

### Chunk ID 设计

扁平序号格式，支持 Agent 通过 ±N 获取相邻 chunk：

```
{doc_prefix}_L1_0000   ← 第1章
{doc_prefix}_L2_0000   ← 第1个小节
{doc_prefix}_L3_0000   ← 第1个知识块
{doc_prefix}_L3_0001   ← 第2个知识块（相邻）
{doc_prefix}_L4_0000   ← 第1个证据块
```

## 检索管线

### 智能路由

根据查询特征自动选择最优检索路径：

```
用户查询
   │
   ├─ HyDE 开启 → LLM 生成假想文档 → 直接 L3/L4 检索
   │
   ├─ 短查询（≤15字）→ 双阶段检索
   │     └─ Stage 1: L2 粗定位 → Stage 2: 范围限定 L3/L4 细召回
   │
   └─ 长查询 → 直接 L3/L4 全局检索
```

### 检索流程

1. **Query Rewriting**（可选）：LLM 改写查询，增加语义关键词
2. **HyDE**（可选）：LLM 生成假想答案文本，用其向量检索，适合口语化查询
3. **Hybrid Search**：Dense（语义向量）+ BM25（稀疏关键词）双路检索，RRF 融合
4. **Rerank**：qwen3-rerank 精排，top10 召回 → top3 精选
5. **Multi-layer Context**：命中 chunk → 回溯 L1/L2/L3 祖先节点，组装完整上下文
6. **Pack Expansion**：命中 chunk 属于已知知识包时，自动扩展相关 chunk（见下文）

### 质量自适应

RAG Tool 支持 `mode=auto`：当 rerank 置信度低于阈值时，自动升级为 HyDE 模式重检索。

## 知识包学习（Knowledge Pack）

系统会从对话中学习，自动记录有效的 chunk 组合，让检索越来越聪明。

### 原理

```
对话 N: Agent 检索 → 多个 chunk 进入 LLM 上下文 → 同一 L2 下的 chunk 被记录为知识包
对话 N+1: 新查询命中某个 chunk → 自动扩展知识包中的其他 chunk → Agent 获得完整上下文
```

### 机制

- **自动学习**：当同一 L2 section 下有 2+ 个 chunk 共同进入最终 LLM 上下文时，自动形成知识包
- **智能去重**：chunk 重叠 > 80% 的包自动合并，同一 L2 最多 5 个包
- **自然衰减**：长期未命中的包自动清理，命中过的包自动强化
- **文档隔离**：文档重新入库时自动清除相关包

## 安全审计

内置两阶段安全检查，防止 prompt 注入、token 滥用和话题偏移。

### 检测策略

| 阶段 | 方法 | 说明 |
|------|------|------|
| Stage 1 | 正则/关键词规则 | 零延迟拦截 prompt 注入、token 滥用（π位数/数学猜想）、图片滥用 |
| Stage 2 | LLM 轻量分类 | 语义级检测话题偏移，max_tokens=10，仅在规则未命中时触发 |

### 防护范围

- **Prompt 注入**：中英文双语检测（忽略指令、角色切换、jailbreak 等）
- **Token 滥用**：数学猜想证明、超长数字列出、大规模计算请求
- **话题偏移**：娱乐八卦、游戏攻略、算命占卜等无关话题
- **图片滥用**：访问非知识库资源、路径遍历攻击
- **系统 prompt 保护**：在 system prompt 中注入安全边界指令

## Web 界面

基于 FastAPI + SSE 流式传输，提供 ChatGPT/DeepSeek 风格的对话界面。

### 功能

- **流式对话**：逐 token 实时输出，SSE 推送
- **思考过程展示**：tool 调用期间展开显示（spinner + 工具名 + 参数），收到回答后自动折叠为 "▶ 思考过程 (N 步)"
- **中间推理**：LLM 在 tool 调用之间的推理文本以紫色底色显示在思考步骤内
- **文件上传**：支持 PDF / Markdown，带进度提示（MinerU 转换 → 知识库载入 → 向量化入库）
- **知识库管理**：`data/knowledge.md` 记录已入库课本列表
- **多轮对话**：左侧边栏管理多个对话
- **消息操作**：复制回答、返回修改、继续生成（递归限制时）
- **移动端适配**：响应式布局，手机端侧边栏滑出

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/api/chat` | POST | SSE 流式聊天（`event: thinking/token/tool_result/done/recursion_limit`） |
| `/api/upload` | POST | 文件上传入库 |
| `/api/knowledge` | GET | 获取已入库课本列表 |
| `/api/images/{path}` | GET | 知识库图片服务（路径遍历防护） |
| `/api/conversations` | GET | 对话列表 |
| `/api/conversations/{id}` | GET/DELETE | 获取/删除对话 |

## Agent 工具

| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `knowledge_base_search` | 知识库检索 | `query`, `mode`(auto/standard/hyde), `limit`, `advanced` |
| `fetch_neighbor_context` | 获取相邻 chunk 上下文 | `chunk_id`, `n_before`, `n_after` |
| `view_image` | 查看知识库引用的图片 | `image_path` |
| `web_search` | 互联网搜索（智谱 API） | `query` |

## 项目结构

```
src/agent/
├── server.py                   # FastAPI Web 服务 + SSE 流式接口 + 安全检查
├── graph.py                    # LangGraph Agent 定义 + 系统 prompt
├── agent.py                    # CLI 交互式对话入口
├── persistence.py              # SQLite 对话持久化
├── rag/
│   ├── Loader/
│   │   ├── base_loader.py      # 加载器基类
│   │   ├── md_loader.py        # Markdown 解析（TOC、标题分类、分块）
│   │   └── minerU.py           # MinerU API（PDF → Markdown）
│   ├── splitter/
│   │   ├── base_splitter.py    # 分块器基类
│   │   └── md_splitter.py      # Markdown 四层分块器
│   ├── data_embedding.py       # 入库管线：chunk → embedding → Milvus
│   ├── milvus_manage.py        # Milvus 客户端
│   ├── retriever.py            # 检索管线：路由 → 混合检索 → 重排序 → Pack 扩展
│   └── knowledge_pack.py       # 知识包学习：记录、去重、衰减、扩展
├── security/
│   ├── __init__.py             # 导出 check_message
│   ├── guard.py                # 两阶段安全检查（规则 + LLM）
│   └── patterns.py             # 正则模式、关键词列表
├── models/
│   ├── chat_model.py           # 对话模型（多 Provider）
│   ├── embedding_model.py      # Embedding 模型
│   └── reranker_model.py       # 重排序模型
├── tools/
│   ├── rag_tool.py             # 知识库检索工具
│   ├── context_tool.py         # 邻居上下文工具
│   ├── image_tool.py           # 图片查看工具（路径白名单校验）
│   └── web_tool.py             # Web 搜索工具
├── utils/
│   ├── logger_handler.py       # 日志
│   └── path_handler.py         # 路径处理
└── schema/
    └── tools_schema.py         # 工具 Schema

static/
└── index.html                  # Web 前端（单文件，内联 CSS/JS）

data/
├── chat.db                     # SQLite 对话数据库（含 knowledge_packs 表）
├── processed_md5.txt           # 已入库文档去重记录
├── knowledge.md                # 已入库课本列表
└── ...                         # 上传的文档和 MinerU 输出
```

## Milvus Schema（16 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 (PK) | 自增主键 |
| `text_dense` | FLOAT_VECTOR | 语义向量 |
| `text_sparse` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量（自动由 BM25 Function 生成） |
| `text` | VARCHAR(16384) | chunk 清洁文本（已剥离图片语法） |
| `doc_id` | VARCHAR | 文档唯一 ID（内容指纹） |
| `filename` | VARCHAR | 文件名 |
| `file_path` | VARCHAR | 文件路径 |
| `chunk_id` | VARCHAR | chunk 唯一 ID |
| `parent_chunk_id` | VARCHAR | 父 chunk ID |
| `root_chunk_id` | VARCHAR | 根 chunk ID（L1） |
| `chunk_level` | INT64 | 层级（1-4） |
| `title_path` | VARCHAR | 标题路径 |
| `title` | VARCHAR | 当前标题 |
| `content_type` | VARCHAR | 内容类型（chapter/section/...） |
| `page_number` | INT64 | 页码 |
| `image_paths` | VARCHAR(4096) | 图片路径（分号分隔） |

## 环境配置

### 依赖

```bash
conda create -n study python=3.12
conda activate study
pip install -e . "langgraph-cli[inmem]"
```

### 环境变量

创建 `.env` 文件：

```env
# MinerU（PDF 解析）
MINERU_API_KEY=your_mineru_api_key
MINERU_BASE_URL=https://mineru.net

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=study_agent

# 模型
DASHSCOPE_API_KEY=your_dashscope_key

# 检索参数（可选，均有默认值）
RETRIEVE_SHORT_QUERY_THRESHOLD=15    # 短查询阈值（字数）
RETRIEVE_HYBRID_LIMIT=15             # 混合检索召回数
RERANK_TOP_N=5                       # 重排序精选数
RERANK_SCORE_THRESHOLD=0.3           # 质量阈值（低于则触发重检索）
RERANK_MODEL=qwen3-rerank            # 重排序模型

# LangSmith（可选）
LANGSMITH_API_KEY=lsv2_...
```

## 快速开始

### 启动 Web 服务

```bash
conda activate study
set PYTHONPATH=src          # Windows
export PYTHONPATH=src       # Linux/Mac

# 启动 FastAPI 服务
uvicorn agent.server:app --reload --host 0.0.0.0 --port 8080

# 浏览器访问 http://localhost:8080
```

### CLI 对话

```bash
python -m agent.agent
```

### LangGraph Studio

```bash
langgraph dev
```

### 编程入库

```python
from agent.rag.data_embedding import RAGPipelineService

service = RAGPipelineService(model_name="dashscope", dimensions=1024)
service.ingest_file(r"data\算法基础\算法基础.md")      # Markdown
service.ingest_file(r"data\教材.pdf")                   # PDF（自动 MinerU 转换）
service.close()
```

### 单独测试检索

```python
from agent.rag.retriever import Retriever

retriever = Retriever(model_name="dashscope", dimensions=1024)
contexts = retriever.retrieve("快速排序的时间复杂度是多少？", limit=3)
for ctx in contexts:
    print(ctx["title_path"], ctx.get("rerank_score"))
```

## 开发

- LangGraph Studio 支持热重载，修改代码后自动生效
- 集成 [LangSmith](https://smith.langchain.com/) 进行链路追踪和性能分析

## License

MIT
