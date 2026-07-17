"""
============================================================
AsyncRAGSystem - FastAPI 应用入口
支持100人异步并发的RAG检索问答系统
============================================================

启动方式:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

架构设计:
    - FastAPI 异步框架 (基于 Starlette + asyncio)
    - 使用 uvicorn 多 worker 模式实现真正的并行处理
    - 每个 worker 内部使用 asyncio 协程处理高并发I/O
    - 4 workers × 协程并发 ≈ 可支持数百并发连接

并发处理链路:
    用户请求 → FastAPI async handler → httpx async → Ollama
                                      → asyncio.to_thread → Milvus
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持直接运行 python app/main.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import init_services, shutdown_services
from app.api.routes import router as api_router

# ============================================================
# 日志配置
# ============================================================

def setup_logging():
    """配置结构化日志输出"""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # 降低第三方库的日志级别 (避免噪音)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


# ============================================================
# 应用生命周期管理 (Lifespan)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器。

    Startup:
        - 初始化 Ollama 连接池 (httpx.AsyncClient)
        - 初始化 Milvus 连接 (MilvusClient)
        - 探测嵌入向量维度并创建/确认 Milvus Collection

    Shutdown:
        - 优雅关闭所有HTTP连接池
        - 释放 Milvus 连接资源
    """
    logger.info("=" * 60)
    logger.info("AsyncRAGSystem 正在启动...")
    logger.info(f"Ollama: {settings.OLLAMA_HOST} (嵌入:{settings.EMBEDDING_MODEL}, LLM:{settings.LLM_MODEL})")
    logger.info(f"Milvus: {settings.milvus_uri}")
    logger.info(f"Redis:  {'已启用' if settings.CACHE_ENABLED else '已禁用'} ({settings.REDIS_HOST}:{settings.REDIS_PORT})")
    logger.info(f"并发配置: workers={settings.WORKERS}, llm_sem={settings.OLLAMA_LLM_MAX_CONCURRENT}, embed_sem={settings.OLLAMA_EMBED_MAX_CONCURRENT}")
    logger.info("=" * 60)

    try:
        # 初始化所有服务 (Ollama客户端、Milvus连接、Collection创建)
        await init_services()
        logger.info("✅ 所有服务初始化完成，系统就绪")
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        logger.warning("系统将以降级模式运行，部分功能不可用")

    yield  # 应用运行期间

    # Shutdown 清理
    logger.info("AsyncRAGSystem 正在关闭...")
    await shutdown_services()
    logger.info("✅ AsyncRAGSystem 已安全关闭")


# ============================================================
# FastAPI 应用实例
# ============================================================

app = FastAPI(
    title="AsyncRAGSystem",
    description="""
## 异步RAG检索问答系统 (v0.2.0)

支持100人异步并发访问的RAG (检索增强生成) 系统。

### 核心特性:
- **异步架构**: 基于 FastAPI + asyncio，全链路异步I/O
- **Nginx 网关**: 反向代理 + least_conn负载均衡 + 限流保护
- **本地模型**: Ollama 部署的 qwen3.5-9b (LLM) + bge-m3 (嵌入)
- **向量存储**: Milvus (HNSW语义索引 + BM25关键词倒排索引)
- **混合检索**: BM25 + 语义向量 RRF融合检索
- **两级缓存**: Redis L1精确缓存 + L2语义缓存, 命中延迟 <5ms
- **高并发**: 连接池 + 多级信号量控制，稳定支撑100+并发
- **流式生成**: 支持 SSE (Server-Sent Events) 实时推送

### API 端点:
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 系统健康检查 |
| GET | `/api/v1/collections/stats` | 向量库统计 |
| GET | `/api/v1/cache/stats` | Redis缓存命中率 |
| POST | `/api/v1/ingest` | 文档摄入 |
| POST | `/api/v1/query` | RAG问答 (混合检索) |
| POST | `/api/v1/query/stream` | 流式RAG问答 (SSE) |
| DELETE | `/api/v1/documents` | 文档删除 |
    """,
    version="0.2.0",
    lifespan=lifespan,
)

# ============================================================
# 中间件配置
# ============================================================

# CORS 跨域中间件 (允许前端应用跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志与耗时统计中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    记录每个HTTP请求的方法、路径、状态码和处理耗时。
    用于性能监控和问题排查。
    """
    import time
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000

    # 仅记录API请求 (跳过静态资源等)
    if request.url.path.startswith("/api"):
        logger.info(
            f"{request.method} {request.url.path} → "
            f"{response.status_code} ({elapsed:.1f}ms)"
        )

    return response


# 请求体大小限制中间件
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """
    限制请求体大小，防止超大请求耗尽内存。
    默认限制: 50MB (可在配置中修改 MAX_UPLOAD_SIZE_MB)。
    """
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")

    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"请求体过大，最大允许 {settings.MAX_UPLOAD_SIZE_MB}MB"
            },
        )

    return await call_next(request)


# ============================================================
# 路由注册
# ============================================================

app.include_router(api_router)


# ============================================================
# 根路径 (欢迎页)
# ============================================================

@app.get("/")
async def root():
    """API 根路径，返回系统基本信息"""
    return {
        "name": "AsyncRAGSystem",
        "version": "0.1.0",
        "description": "异步RAG检索问答系统",
        "docs": "/docs",          # FastAPI 自动生成的 Swagger UI
        "redoc": "/redoc",        # FastAPI 自动生成的 ReDoc
        "health": "/api/v1/health",
    }


# ============================================================
# 直接运行入口 (开发调试用)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"启动开发服务器: http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,          # 开发模式单worker (方便调试)
        reload=True,        # 代码变更自动重载
        log_level="info",
    )
