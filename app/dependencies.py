"""
============================================================
AsyncRAGSystem - 依赖注入模块
管理所有服务实例的生命周期 (单例模式) 以及 FastAPI 依赖注入
============================================================

设计要点:
1. 全局单例服务实例 (避免重复创建连接池)
2. FastAPI Depends() 风格的依赖注入函数
3. 服务懒加载 (首次调用时初始化)
4. 优雅关闭支持 (通过 FastAPI lifespan 管理)
"""

import logging

from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStoreService
from app.services.document import DocumentService
from app.services.retrieval import RetrievalService
from app.services.cache import CacheService

logger = logging.getLogger(__name__)

# ============================================================
# 全局服务单例
# 在应用启动时创建，整个生命周期内复用
# ============================================================

# 这些实例在 lifespan 中初始化
_embedding_service: EmbeddingService | None = None
_llm_service: LLMService | None = None
_vector_store: VectorStoreService | None = None
_document_service: DocumentService | None = None
_retrieval_service: RetrievalService | None = None
_cache_service: CacheService | None = None


# ============================================================
# 服务初始化 (在 FastAPI lifespan startup 中调用)
# ============================================================

async def init_services(embedding_dimension: int | None = None):
    """
    初始化所有服务实例。

    在应用启动时调用一次，建立 Ollama/Milvus/Redis 连接。

    Args:
        embedding_dimension: 嵌入向量维度。若为None则在首次嵌入时自动探测。
    """
    global _embedding_service, _llm_service, _vector_store
    global _document_service, _retrieval_service, _cache_service

    logger.info("正在初始化服务...")

    # 创建基础服务
    _embedding_service = EmbeddingService()
    _llm_service = LLMService()
    _vector_store = VectorStoreService()

    # 初始化 Redis 缓存 (连接失败自动降级, 不影响主流程)
    _cache_service = CacheService()
    await _cache_service.connect()

    # 如果已知向量维度，预先创建/确认 Milvus Collection
    if embedding_dimension:
        await _vector_store.ensure_collection(embedding_dimension)
    else:
        # 自动探测维度
        test_vectors = await _embedding_service.embed_texts(["初始化探测"])
        dim = len(test_vectors[0])
        await _vector_store.ensure_collection(dim)
        logger.info(f"自动探测到嵌入维度: {dim}")

    # 创建编排服务 (将缓存注入检索服务)
    _document_service = DocumentService(
        embedding_service=_embedding_service,
        vector_store=_vector_store,
    )
    _retrieval_service = RetrievalService(
        embedding_service=_embedding_service,
        llm_service=_llm_service,
        vector_store=_vector_store,
        cache_service=_cache_service,     # ← Redis缓存注入
    )

    logger.info("所有服务初始化完成")


async def shutdown_services():
    """关闭所有服务，释放连接资源。在应用关闭时调用。"""
    global _embedding_service, _llm_service, _vector_store
    global _document_service, _retrieval_service, _cache_service

    logger.info("正在关闭服务...")

    if _embedding_service:
        await _embedding_service.close()
    if _llm_service:
        await _llm_service.close()
    if _cache_service:
        await _cache_service.disconnect()

    # MilvusClient 关闭 (可选，连接会自动回收)
    if _vector_store:
        await _vector_store.close()

    _embedding_service = None
    _llm_service = None
    _vector_store = None
    _document_service = None
    _retrieval_service = None
    _cache_service = None

    logger.info("所有服务已关闭")


# ============================================================
# FastAPI 依赖注入函数
# 使用 Depends() 在路由处理函数中获取服务实例
# ============================================================

async def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例 (FastAPI Depends)"""
    if _embedding_service is None:
        raise RuntimeError("EmbeddingService 尚未初始化，请检查应用启动流程")
    return _embedding_service


async def get_llm_service() -> LLMService:
    """获取LLM服务实例 (FastAPI Depends)"""
    if _llm_service is None:
        raise RuntimeError("LLMService 尚未初始化，请检查应用启动流程")
    return _llm_service


async def get_vector_store() -> VectorStoreService:
    """获取向量存储服务实例 (FastAPI Depends)"""
    if _vector_store is None:
        raise RuntimeError("VectorStoreService 尚未初始化，请检查应用启动流程")
    return _vector_store


async def get_document_service() -> DocumentService:
    """获取文档处理服务实例 (FastAPI Depends)"""
    if _document_service is None:
        raise RuntimeError("DocumentService 尚未初始化，请检查应用启动流程")
    return _document_service


async def get_retrieval_service() -> RetrievalService:
    """获取RAG检索服务实例 (FastAPI Depends)"""
    if _retrieval_service is None:
        raise RuntimeError("RetrievalService 尚未初始化，请检查应用启动流程")
    return _retrieval_service


async def get_cache_service() -> CacheService:
    """获取缓存服务实例 (FastAPI Depends)"""
    if _cache_service is None:
        raise RuntimeError("CacheService 尚未初始化，请检查应用启动流程")
    return _cache_service
