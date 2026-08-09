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
from typing import Any, Dict, List
import os
import uuid

from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.core.document_loader import DocumentLoader
from app.core.preprocess import DocumentPreprocessor

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

        self.doc_loader = DocumentLoader()
        self.doc_preprocessor = DocumentPreprocessor()

    async def ingest_texts(
        self,
        source_path: str,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        摄入文本文档的完整流水线。

        处理流程:
        1. 文本分块 (RecursiveCharacterTextSplitter)
        2. 批量向量化 (调用 Ollama 嵌入API)
        3. 存入 Milvus

        Args:
            source_path: 文件路径或文件夹路径。
            batch_size: 嵌入批处理大小 (避免单次请求过大)。

        Returns:
            包含处理统计的字典: {document_count, chunk_count, chunk_ids}
        """
        
        if os.path.isdir(source_path):
            documents = await self.doc_loader.load_directory(source_path)
        else:
            documents = await self.doc_loader.load_file(source_path)

        if not documents:
            logger.warning("未加载到任何文档")
            return {"status": "empty", "documents": 0}

        # --- Phase 2: 预处理（清洗 + 分块） ---
        all_chunks, clean_stats = self.doc_preprocessor.process(documents)
        logger.info(
            f"Phase 2 完成: 清洗后 {len(all_chunks)} 个文档片段 "
            f"(移除 {clean_stats.removed_docs} 个低质量片段)"
        )

        if not all_chunks:
            return {"document_count": len(documents), "chunk_count": 0, "insert_count": 0}

        # === Step 2: 批量嵌入 + 存储 ===
        all_insert_count = 0
        # 分批处理，避免单次嵌入请求过大
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_texts = [chunk.page_content for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            batch_doc_ids = [chunk.metadata.get("doc_id", "") for chunk in batch]
            batch_file_names = [chunk.metadata.get("source_file", "") for chunk in batch]
            batch_chunk_index = [chunk.metadata.get("chunk_index", 0) for chunk in batch]

            # 向量化 (Ollama 批量嵌入)
            vectors = await self._embedding.embed_texts(batch_texts)

            # 存入 Milvus
            insert_count = await self._vector_store.upsert(
                vectors=vectors,
                texts=batch_texts,
                doc_ids=batch_doc_ids,
                file_names=batch_file_names,
                metadatas=batch_metadatas,
                chunk_index=batch_chunk_index,
                batch_id = i,
            )
            all_insert_count += insert_count

            logger.info(
                f"批次 {i // batch_size + 1}: {len(batch)} 块已嵌入并存储"
            )

        result = {
            "document_count": len(documents),
            "chunk_count": len(all_chunks),
            "insert_count": all_insert_count,
        }

        logger.info(
            f"文档摄入完成: {result['document_count']} 篇文档, "
            f"{result['insert_count']} 个文本块已存入 Milvus"
        )

        # =========================
        # Phase 3: 清理多余旧切片
        # =========================
        total_deleted = 0
        # 计算每个文档的切片总数（chunk_index 最大值 + 1）
        doc_total_chunks = {}
        for chunk in all_chunks:
            doc_id = chunk.metadata["doc_id"]
            idx = chunk.metadata["chunk_index"]
            doc_total_chunks[doc_id] = max(doc_total_chunks.get(doc_id, 0), idx + 1)

        for doc_id, count in doc_total_chunks.items():
            safe_doc_id = doc_id.replace("\\", "/")   # 或者直接替换为双反斜杠: doc_id.replace("\\", "\\\\")
            filter_expr = f'doc_id == "{safe_doc_id}" and chunk_index >= {count}'
            deleted = await self._vector_store.delete_by_filter(filter_expr)
            total_deleted += deleted
             
        logger.info(f"清理阶段完成：共删除 {total_deleted} 条旧切片")

        result = {
            "document_count": len(documents),
            "chunk_count": len(all_chunks),
            "insert_count": all_insert_count,
            "deleted_old_chunks": total_deleted,   # 可选，方便监控
        }

        return result

    async def delete_chunks(self, filter_expr: str) -> int:
        """
        删除指定的文档块。

        Args:
            chunk_ids: 要删除的chunk ID列表。

        Returns:
            实际删除的数量。
        """
        return await self._vector_store.delete_by_filter(filter_expr)
