"""
============================================================
AsyncRAGSystem - API 数据模型 (Pydantic Schemas)
定义所有请求/响应的数据结构，自动校验与文档生成
============================================================
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ==================== 文档摄入相关 ====================

class DocumentInput(BaseModel):
    """
    文档摄入请求体。
    支持直接传入文本或文本列表。
    """
    texts: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="待摄入的文本列表，每项为一个独立文档",
        examples=[["这是一段文档内容。", "这是另一段文档内容。"]]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="附加到所有文档的公共元数据 (如来源、作者等)"
    )


class IngestResponse(BaseModel):
    """文档摄入响应"""
    success: bool = Field(..., description="操作是否成功")
    document_count: int = Field(..., description="摄入的文档数量")
    chunk_count: int = Field(..., description="切分后的文本块总数")
    message: str = Field(..., description="操作结果描述")


# ==================== RAG 问答相关 ====================

class QueryInput(BaseModel):
    """
    RAG 问答请求体。
    """
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="用户提出的问题",
        examples=["什么是RAG系统？"]
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="检索返回的文档片段数量 (默认使用全局配置)"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0,
        description="LLM生成温度 (默认使用全局配置)"
    )
    stream: bool = Field(
        default=False,
        description="是否使用流式生成 (SSE)"
    )


class SourceDocument(BaseModel):
    """检索到的源文档片段"""
    chunk_id: str = Field(..., description="文本块唯一标识")
    text: str = Field(..., description="文本块内容")
    score: float = Field(..., description="相似度分数 (0~1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="文档元数据"
    )


class QueryResponse(BaseModel):
    """RAG 问答响应"""
    answer: str = Field(..., description="LLM生成的回答")
    sources: List[SourceDocument] = Field(
        default_factory=list, description="检索到的源文档片段"
    )
    processing_time_ms: float = Field(
        ..., description="总处理耗时 (毫秒)"
    )
    model: str = Field(..., description="使用的LLM模型名称")
    cached: bool = Field(
        default=False, description="是否来自Redis缓存命中"
    )
    search_method: str = Field(
        default="hybrid(RRF)", description="检索方法: hybrid(RRF) / dense(fallback)"
    )


# ==================== 系统状态相关 ====================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: healthy / degraded / unhealthy")
    version: str = Field(default="0.2.0", description="系统版本号")
    ollama_connected: bool = Field(..., description="Ollama 连接状态")
    milvus_connected: bool = Field(..., description="Milvus 连接状态")
    embedding_model: str = Field(..., description="当前使用的嵌入模型")
    llm_model: str = Field(..., description="当前使用的LLM模型")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="检查时间戳"
    )


class CollectionStats(BaseModel):
    """Milvus 集合统计信息"""
    collection_name: str = Field(..., description="集合名称")
    document_count: int = Field(..., description="存储的文档块总数")
    exists: bool = Field(..., description="集合是否存在")


# ==================== 文档删除相关 ====================

class DeleteRequest(BaseModel):
    """文档删除请求"""
    chunk_ids: Optional[List[str]] = Field(
        default=None, description="要删除的文本块ID列表"
    )
    filter_expr: Optional[str] = Field(
        default=None, description="Milvus过滤表达式 (如 source == 'web')"
    )


class DeleteResponse(BaseModel):
    """文档删除响应"""
    success: bool = Field(..., description="操作是否成功")
    deleted_count: int = Field(..., description="删除的文档块数量")


# ==================== 缓存统计相关 ====================

class CacheStatsResponse(BaseModel):
    """Redis 缓存统计响应"""
    enabled: bool = Field(..., description="缓存是否启用且连接正常")
    hits_exact: int = Field(..., description="L1精确缓存命中次数")
    hits_semantic: int = Field(..., description="L2语义缓存命中次数")
    misses: int = Field(..., description="缓存未命中次数")
    hit_rate: float = Field(..., description="缓存命中率 (0~1)")
