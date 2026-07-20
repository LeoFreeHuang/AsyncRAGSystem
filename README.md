# AsyncRAGSystem - 异步RAG检索问答系统

支持 **100人异步并发访问** 的 RAG (Retrieval-Augmented Generation) 检索增强生成系统。

> 📖 **详细架构设计文档**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 架构概览

```
用户 → Nginx(网关) → FastAPI(服务) → Redis(缓存)
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
         Embedding    Hybrid Search    LLM
         (Ollama)   (Milvus BM25+向量) (Ollama)
```

## 核心特性 (v0.2.0)

| 特性 | 说明 |
|------|------|
| **🔀 Nginx 网关** | 反向代理 + 负载均衡 (least_conn) + 限流 (100r/s) |
| **💾 Redis 两级缓存** | L1精确缓存 + L2语义缓存, 命中率20~40%, 延迟 <5ms |
| **🔍 BM25 + 向量混合检索** | Milvus 内置 BM25 + HNSW语义检索 + RRF融合排序 |
| **⚡ 异步架构** | FastAPI + asyncio 全链路异步 I/O |
| **🤖 本地模型** | Ollama qwen3.5:9b (LLM) + bge-m3 (嵌入) |

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 网关 | **Nginx** | 反向代理、负载均衡、限流 |
| Web框架 | **FastAPI** | 异步高性能 Python Web 框架 |
| 缓存 | **Redis** | 两级缓存 (精确+语义) |
| LLM | **qwen3.5:9b (4bit)** | Ollama 本地部署 |
| 嵌入 | **bge-m3** | Ollama 本地部署, 1024维 |
| 向量库 | **Milvus** | BM25 + HNSW 混合检索 |
| HTTP | **httpx** | 异步连接池 |

## 项目结构

```
AsyncRAGSystem/
├── nginx/
│   └── nginx.conf              # Nginx 网关配置
├── docs/
│   └── ARCHITECTURE.md         # 详细架构设计文档
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 全局配置
│   ├── dependencies.py         # 依赖注入
│   ├── api/
│   │   ├── routes.py           # API 路由 (含缓存统计)
│   │   └── schemas.py          # 数据模型
│   ├── services/
│   │   ├── embedding.py        # 嵌入服务
│   │   ├── llm.py              # LLM服务
│   │   ├── vector_store.py     # Milvus (BM25+混合检索)
│   │   ├── cache.py            # Redis 缓存服务
│   │   ├── document.py         # 文档摄入
│   │   └── retrieval.py        # RAG流水线
│   └── core/
│       └── chunking.py         # 文本分块
├── scripts/
│   └── init_milvus.py          # Milvus 初始化
├── .env / .env.example
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 拉取模型
ollama pull qwen3.5:9b
ollama pull bge-m3

# 启动 Milvus (Docker)
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# 启动 Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 2. 安装与配置

```bash
pip install -r requirements.txt
cp .env.example .env   # 编辑配置
python scripts/init_milvus.py --reset  # 初始化 Milvus (含BM25 Schema)
```

### 3. 启动

```bash
# 开发模式
python -m app.main

# 生产模式 (4 workers)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4

# Windows下启用Nginx 网关 
在Nginx安装根目录执行: start nginx
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 摄入文档
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["RAG是检索增强生成技术，结合了信息检索和文本生成。"]}'

# RAG问答 (混合检索 + 自动缓存)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG?"}'

# 缓存统计
curl http://localhost:8000/api/v1/cache/stats

#验证请求是否由Nginx转发
curl http://localhost:8080/api/v1/health
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/collections/stats` | Milvus统计 |
| `GET` | `/api/v1/cache/stats` | **Redis缓存命中率** |
| `POST` | `/api/v1/ingest` | 文档摄入 |
| `POST` | `/api/v1/query` | **混合检索RAG问答** |
| `POST` | `/api/v1/query/stream` | SSE流式问答 |
| `DELETE` | `/api/v1/documents` | 删除文档 |

访问 `http://localhost:8000/docs` 查看完整 Swagger API 文档。

