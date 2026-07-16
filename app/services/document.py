"""
============================================================
AsyncRAGSystem - 文档处理服务 (Document Service)
负责文档的摄入、分块、向量化和存储的完整流程编排
============================================================

设计要点:
1. 编排 "分块 → 嵌入 → 存储" 的完整摄入流水线
2. 批量处理优化: 累积到一定数量后批量调用嵌入API
3. 异步流水线: 各阶段可并行处理不同批次
"""

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.chunking import RecursiveCharacterTextSplitter
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    文档处理服务。

    负责将原始文档经过分块、向量化后存入Milvus。
    这是RAG系统的"写入"路径的核心组件。

    使用示例:
        doc_service = DocumentService(embedding_service, vector_store)
        result = await doc_service.ingest_texts(["文档内容1", "文档内容2"])
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ):
        """
        Args:
            embedding_service: 嵌入服务实例。
            vector_store: 向量存储服务实例。
        """
        self._embedding = embedding_service
        self._vector_store = vector_store

        # 文本分块器 (使用全局配置的chunk_size和overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    async def ingest_texts(
        self,
        texts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        摄入文本文档的完整流水线。

        处理流程:
        1. 文本分块 (RecursiveCharacterTextSplitter)
        2. 批量向量化 (调用 Ollama 嵌入API)
        3. 存入 Milvus

        Args:
            texts: 原始文本列表 (每条为一个独立文档)。
            metadata: 附加到所有文档的公共元数据。
            batch_size: 嵌入批处理大小 (避免单次请求过大)。

        Returns:
            包含处理统计的字典: {document_count, chunk_count, chunk_ids}
        """
        if not texts:
            return {"document_count": 0, "chunk_count": 0, "chunk_ids": []}

        logger.info(f"开始摄入文档: {len(texts)} 篇")

        # === Step 1: 文本分块 ===
        raw_documents = [
            {"text": text, "metadata": metadata or {}}
            for text in texts
        ]
        all_chunks = self._splitter.split_documents(raw_documents)
        logger.info(f"分块完成: {len(texts)} 篇文档 → {len(all_chunks)} 个文本块")

        if not all_chunks:
            return {"document_count": len(texts), "chunk_count": 0, "chunk_ids": []}

        # === Step 2: 批量嵌入 + 存储 ===
        all_chunk_ids = []

        # 分批处理，避免单次嵌入请求过大
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            batch_metadatas = [chunk["metadata"] for chunk in batch]

            # 向量化 (Ollama 批量嵌入)
            vectors = await self._embedding.embed_texts(batch_texts)

            # 存入 Milvus
            chunk_ids = await self._vector_store.insert(
                vectors=vectors,
                texts=batch_texts,
                metadatas=batch_metadatas,
            )
            all_chunk_ids.extend(chunk_ids)

            logger.info(
                f"批次 {i // batch_size + 1}: {len(batch)} 块已嵌入并存储"
            )

        result = {
            "document_count": len(texts),
            "chunk_count": len(all_chunks),
            "chunk_ids": all_chunk_ids,
        }

        logger.info(
            f"文档摄入完成: {result['document_count']} 篇文档, "
            f"{result['chunk_count']} 个文本块已存入 Milvus"
        )
        return result

    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        删除指定的文档块。

        Args:
            chunk_ids: 要删除的chunk ID列表。

        Returns:
            实际删除的数量。
        """
        return await self._vector_store.delete_by_ids(chunk_ids)
