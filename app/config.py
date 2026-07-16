"""
============================================================
AsyncRAGSystem - 全局配置管理
使用 pydantic-settings 从环境变量 / .env 文件加载配置
============================================================
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    应用全局配置类。
    所有配置项均可通过环境变量或 .env 文件覆盖，优先级: 环境变量 > .env > 默认值
    """

    # ==================== 服务端配置 ====================
    HOST: str = Field(default="0.0.0.0", description="FastAPI 监听地址")
    PORT: int = Field(default=8000, description="FastAPI 监听端口")
    WORKERS: int = Field(default=4, description="Uvicorn worker 进程数")

    # ==================== Ollama 配置 ====================
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434", description="Ollama 服务地址"
    )
    EMBEDDING_MODEL: str = Field(
        default="bge-m3", description="嵌入模型名称 (需已在Ollama中拉取)"
    )
    LLM_MODEL: str = Field(
        default="qwen3.5:9b", description="大语言模型名称 (需已在Ollama中拉取)"
    )

    # ==================== Milvus 向量数据库配置 ====================
    MILVUS_HOST: str = Field(default="localhost", description="Milvus 服务主机")
    MILVUS_PORT: int = Field(default=19530, description="Milvus 服务端口")
    MILVUS_TOKEN: str = Field(
        default="root:Milvus", description="Milvus 认证令牌 (user:password)"
    )
    MILVUS_COLLECTION: str = Field(
        default="rag_documents", description="Milvus 集合名称"
    )
    # Milvus 连接池配置
    MILVUS_MAX_CONCURRENCY: int = Field(
        default=50, description="Milvus 最大并发操作数 (控制对Milvus的压力)"
    )
    # Milvus 文本分析器 (BM25 需要): standard / english / chinese
    MILVUS_ANALYZER: str = Field(
        default="chinese", description="Milvus 文本分析器类型 (用于BM25分词)"
    )

    # ==================== BM25 混合检索配置 ====================
    # 混合检索权重: dense_weight + sparse_weight = 1.0
    HYBRID_DENSE_WEIGHT: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="混合检索中语义向量权重 (0~1), BM25权重=1-dense_weight"
    )
    # RRF (Reciprocal Rank Fusion) 参数 k
    HYBRID_RRF_K: int = Field(
        default=60, description="RRF融合算法的k参数 (通常为60)"
    )

    # ==================== Redis 缓存配置 ====================
    REDIS_HOST: str = Field(default="localhost", description="Redis 服务主机")
    REDIS_PORT: int = Field(default=6379, description="Redis 服务端口")
    REDIS_DB: int = Field(default=0, description="Redis 数据库编号")
    REDIS_PASSWORD: str = Field(default="", description="Redis 密码 (无密码则留空)")
    # 精确匹配缓存 TTL (秒)
    CACHE_EXACT_TTL: int = Field(
        default=3600, description="精确关键字缓存过期时间 (秒), 默认1小时"
    )
    # 语义缓存 TTL (秒)
    CACHE_SEMANTIC_TTL: int = Field(
        default=1800, description="语义向量缓存过期时间 (秒), 默认30分钟"
    )
    # 是否启用缓存
    CACHE_ENABLED: bool = Field(
        default=True, description="是否启用Redis缓存"
    )

    # ==================== Nginx 反向代理配置 ====================
    # Nginx 上游服务器列表 (逗号分隔, 如 localhost:8000,localhost:8001)
    NGINX_UPSTREAM_SERVERS: str = Field(
        default="localhost:8000",
        description="Nginx upstream FastAPI节点列表 (逗号分隔)"
    )
    NGINX_RATE_LIMIT_RPS: int = Field(
        default=100, description="Nginx 单IP限流 (请求/秒)"
    )

    # ==================== 文档分块配置 ====================
    CHUNK_SIZE: int = Field(default=512, description="文本分块大小 (字符数)")
    CHUNK_OVERLAP: int = Field(
        default=128, description="相邻分块重叠字符数 (保持上下文连贯)"
    )

    # ==================== 检索配置 ====================
    TOP_K: int = Field(default=5, description="检索返回的Top-K相关文档数")
    SIMILARITY_THRESHOLD: float = Field(
        default=0.3, description="相似度阈值, 低于此值的结果将被过滤"
    )

    # ==================== LLM 生成配置 ====================
    LLM_TEMPERATURE: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM生成温度 (越低越确定性)"
    )
    LLM_MAX_TOKENS: int = Field(default=2048, description="LLM最大生成token数")
    LLM_TIMEOUT: int = Field(default=120, description="LLM请求超时时间 (秒)")

    # ==================== Ollama 并发控制 ====================
    # Ollama在单GPU上实际只能串行推理，此信号量控制同时发往Ollama的请求数
    OLLAMA_LLM_MAX_CONCURRENT: int = Field(
        default=3, description="LLM最大并发请求数 (Ollama串行推理, 过大无益)"
    )
    OLLAMA_EMBED_MAX_CONCURRENT: int = Field(
        default=8, description="嵌入模型最大并发请求数"
    )

    # ==================== 请求体大小限制 ====================
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50, description="单次上传文件最大大小 (MB)"
    )

    @property
    def milvus_uri(self) -> str:
        """构建 Milvus 连接 URI"""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    @property
    def ollama_embed_url(self) -> str:
        """Ollama 嵌入 API 端点"""
        return f"{self.OLLAMA_HOST}/api/embed"

    @property
    def ollama_generate_url(self) -> str:
        """Ollama 生成 API 端点"""
        return f"{self.OLLAMA_HOST}/api/generate"

    @property
    def redis_url(self) -> str:
        """构建 Redis 连接 URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    model_config = {
        "env_file": ".env",          # 自动加载 .env 文件
        "env_file_encoding": "utf-8",
        "case_sensitive": True,       # 环境变量名区分大小写
    }


# 全局单例配置对象
settings = Settings()
