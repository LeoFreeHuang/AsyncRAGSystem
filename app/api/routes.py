"""
============================================================
AsyncRAGSystem - API 路由定义
提供 RESTful API 端点: 文档摄入、RAG问答、系统管理
============================================================

API 端点一览:
    GET  /health              - 系统健康检查
    GET  /collections/stats   - Milvus集合统计
    POST /ingest              - 文档摄入 (文本分块→向量化→存储)
    POST /query               - RAG问答 (检索+生成)
    POST /query/stream        - RAG流式问答 (SSE)
    DELETE /documents         - 删除文档块
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    DocumentInput,
    IngestResponse,
    QueryInput,
    QueryResponse,
    HealthResponse,
    CollectionStats,
    DeleteRequest,
    DeleteResponse,
    CacheStatsResponse,
)
from app.dependencies import (
    get_document_service,
    get_retrieval_service,
    get_embedding_service,
    get_vector_store,
    get_cache_service,
)
from app.services.document import DocumentService
from app.services.retrieval import RetrievalService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.cache import CacheService
from app.config import settings

logger = logging.getLogger(__name__)

# 创建路由汇总
router = APIRouter(prefix="/api/v1", tags=["RAG System"])


# ============================================================
# 系统管理端点
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
   
):
    """
    系统健康检查。

    检查 Ollama 和 Milvus 的连接状态，
    用于负载均衡器的健康探测和监控告警。

    Returns:
        HealthResponse: 各组件连接状态和系统信息。
    """
    
    # 检查用户请求是否经过Nginx转发
    nginx_ok = False
    client_host = request.client.host if request.client else None
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded is not None:
        nginx_ok = True
    if nginx_ok:
        logger.info (f"用户请求已通过Nginx转发, Nginx服务正常: (client_host: {client_host}, X-Forwarded-For: {forwarded})")
    else:
        logger.warning(f"用户请求未通过Nginx转发")

    # 检查 Ollama 连接
    ollama_ok = False
    try:
        # 发送一条简单的嵌入请求验证 Ollama 可用
        _ = await embedding_service.embed_texts(["健康检查"])
        ollama_ok = True
    except Exception as e:
        logger.warning(f"Ollama 健康检查失败: {e}")

    # 检查 Milvus 连接
    milvus_ok = False
    try:
        info = await vector_store.get_collection_info()
        milvus_ok = info.get("exists", False)
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")

    # 综合状态判定
    if ollama_ok and milvus_ok:
        status = "healthy"
    elif ollama_ok or milvus_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="0.2.0",
        ollama_connected=ollama_ok,
        milvus_connected=milvus_ok,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
        timestamp=datetime.now(),
    )


@router.get("/collections/stats", response_model=CollectionStats)
async def get_collection_stats(
    vector_store: VectorStoreService = Depends(get_vector_store),
):
    """
    获取 Milvus 向量集合的统计信息。

    Returns:
        CollectionStats: 集合名称、状态、文档数量。
    """
    info = await vector_store.get_collection_info()
    return CollectionStats(**info)


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service),
):
    """
    获取 Redis 缓存命中率统计。

    Returns:
        CacheStatsResponse: 缓存启用状态、各级命中/未命中数、命中率。
    """
    stats = cache_service.get_stats()
    return CacheStatsResponse(**stats)


# ============================================================
# 文档摄入端点
# ============================================================

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_documents(
    doc_input: DocumentInput,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    摄入文本文档到知识库。

    将原始文本经过分块、向量化后存储到 Milvus 向量数据库。
    支持批量摄入多条文本。

    处理流程: 文本分块 → 嵌入向量化 → 存入Milvus

    Args:
        doc_input: 包含文本列表和可选元数据。

    Returns:
        IngestResponse: 摄入统计 (文档数、文本块数)。

    Raises:
        HTTPException 500: 摄入过程发生内部错误。
    """
    try:
        result = await document_service.ingest_texts(
            texts=doc_input.texts,
            metadata=doc_input.metadata,
        )

        return IngestResponse(
            success=True,
            document_count=result["document_count"],
            chunk_count=result["chunk_count"],
            message=(
                f"成功摄入 {result['document_count']} 篇文档, "
                f"生成 {result['chunk_count']} 个文本块"
            ),
        )

    except Exception as e:
        logger.exception("文档摄入失败")
        raise HTTPException(
            status_code=500,
            detail=f"文档摄入失败: {str(e)}",
        )


# ============================================================
# RAG 问答端点
# ============================================================

@router.post("/query", response_model=QueryResponse)
async def rag_query(
    query_input: QueryInput,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """
    RAG 检索增强问答 (非流式)。

    完整的RAG流水线: 查询嵌入 → 向量检索 → 上下文构建 → LLM生成

    Args:
        query_input: 用户问题和可选参数 (top_k, temperature, stream)。

    Returns:
        QueryResponse: 包含答案、检索来源和处理耗时。

    Raises:
        HTTPException 500: RAG流水线执行失败。
    """
    # 流式请求应使用 /query/stream 端点
    if query_input.stream:
        raise HTTPException(
            status_code=400,
            detail="流式请求请使用 /api/v1/query/stream 端点",
        )

    try:
        result = await retrieval_service.query(
            question=query_input.question,
            top_k=query_input.top_k,
            temperature=query_input.temperature,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            processing_time_ms=result["processing_time_ms"],
            model=result["model"],
            cached=result.get("cached", False),
            search_method=result.get("search_method", "hybrid(RRF)"),
        )

    except Exception as e:
        logger.exception("RAG查询失败")
        raise HTTPException(
            status_code=500,
            detail=f"RAG查询失败: {str(e)}",
        )


@router.post("/query/stream")
async def rag_query_stream(
    query_input: QueryInput,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """
    RAG 检索增强问答 (流式 SSE)。

    检索完成后，LLM生成结果以 Server-Sent Events 格式逐token推送，
    适合前端实现打字机效果。

    响应格式: text/event-stream
    每行格式: data: {"token": "生成的文本"}

    Args:
        query_input: 用户问题 (stream字段将被忽略，此端点始终流式)。

    Returns:
        StreamingResponse: SSE 格式的流式生成结果。
    """
    import json

    async def event_generator():
        """SSE 事件生成器"""
        try:
            async for token in retrieval_service.query_stream(
                question=query_input.question,
                top_k=query_input.top_k,
                temperature=query_input.temperature,
            ):
                # SSE 格式: data: <json>\n\n
                event_data = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.exception("流式RAG查询失败")
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲 (如有反向代理)
        },
    )


# ============================================================
# 文档管理端点
# ============================================================

@router.delete("/documents", response_model=DeleteResponse)
async def delete_documents(
    delete_req: DeleteRequest,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    删除指定的文档块。

    支持两种删除方式:
    1. 按 chunk_id 列表精确删除
    2. 按 Milvus 过滤表达式批量删除

    Args:
        delete_req: 包含要删除的chunk_id列表或过滤表达式。

    Returns:
        DeleteResponse: 删除操作结果。
    """
    if not delete_req.chunk_ids and not delete_req.filter_expr:
        raise HTTPException(
            status_code=400,
            detail="请提供 chunk_ids 或 filter_expr 参数",
        )

    try:
        if delete_req.chunk_ids:
            deleted = await document_service.delete_chunks(delete_req.chunk_ids)
        else:
            # 按过滤表达式删除 (未来扩展)
            raise HTTPException(
                status_code=501,
                detail="按filter_expr删除功能待实现",
            )

        return DeleteResponse(
            success=True,
            deleted_count=deleted,
        )

    except Exception as e:
        logger.exception("文档删除失败")
        raise HTTPException(
            status_code=500,
            detail=f"文档删除失败: {str(e)}",
        )
