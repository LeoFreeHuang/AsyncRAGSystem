"""
============================================================
AsyncRAGSystem - 向量存储服务 (Vector Store Service)
基于 Milvus 向量数据库的文档存储与相似性检索
============================================================

设计要点:
1. 使用 PyMilvus 3.0 的 MilvusClient API (官方推荐方式)
2. MilvusClient 是线程安全的，但同步阻塞 → 使用 asyncio.to_thread 包装
3. Semaphore 控制并发数，避免压垮 Milvus
4. 支持动态创建集合、插入向量、相似度搜索、删除等操作
5. HNSW 索引 (高性能近似最近邻搜索) 替代传统的 IVF_FLAT

PyMilvus 3.0.0.0 API 参考:
    - MilvusClient(uri, token) 创建客户端连接
    - client.create_schema() → 创建 Collection Schema
    - client.create_collection() → 创建集合
    - client.insert() → 插入数据
    - client.search() → 向量相似度搜索
    - client.query() → 标量过滤查询
    - client.delete() → 删除数据
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
)

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Milvus 向量存储服务。

    封装所有 Milvus 操作，提供异步接口。
    内置连接池管理和并发控制。

    使用示例:
        store = VectorStoreService()
        await store.ensure_collection()
        ids = await store.insert(vectors, texts, metadatas)
        results = await store.search(query_vector, top_k=5)
    """

    # Milvus Collection Schema 字段名常量
    FIELD_ID = "id"                  # 主键 (VARCHAR, UUID)
    FIELD_TEXT = "text"              # 原始文本内容 (VARCHAR, 启用分词器用于BM25)
    FIELD_DENSE = "dense_vector"     # 语义向量 (FLOAT_VECTOR, 来自 Ollama 嵌入)
    FIELD_SPARSE = "sparse_vector"   # BM25稀疏向量 (SPARSE_FLOAT_VECTOR, Milvus内置BM25生成)

    # BM25 函数名 (Milvus 内部使用)
    BM25_FUNCTION_NAME = "bm25_func"

    def __init__(self):
        """初始化向量存储服务"""
        self._client: MilvusClient | None = None

        # 并发控制: 限制同时进行的Milvus操作数
        self._semaphore = asyncio.Semaphore(settings.MILVUS_MAX_CONCURRENCY)

        # 向量维度 (在 ensure_collection 时确定)
        self._dimension: int | None = None

    def _get_client(self) -> MilvusClient:
        """
        获取 Milvus 客户端 (懒加载单例)。
        MilvusClient 是线程安全的，可以在多线程环境中共享。
        """
        if self._client is None:
            logger.info(f"正在连接 Milvus: {settings.milvus_uri}")
            self._client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.MILVUS_TOKEN,
            )
            logger.info("Milvus 连接成功")
        return self._client

    async def _run_sync(self, func, *args, **kwargs):
        """
        在线程池中运行同步的 Milvus 操作，实现异步化。

        MilvusClient 的所有方法都是同步阻塞的，
        使用 asyncio.to_thread 将其放到线程池执行，避免阻塞事件循环。
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    # ==================== Collection 管理 ====================

    async def ensure_collection(self, dimension: int) -> bool:
        """
        确保 Milvus Collection 存在，若不存在则创建（含BM25混合检索Schema）。

        新 Schema 包含:
          - id (VARCHAR, PK): 文本块的唯一标识
          - text (VARCHAR, enable_analyzer): 原始文本，启用分词器供BM25使用
          - dense_vector (FLOAT_VECTOR): Ollama生成的语义向量
          - sparse_vector (SPARSE_FLOAT_VECTOR): Milvus BM25自动生成的稀疏向量

        索引:
          - HNSW 索引 on dense_vector (COSINE相似度)
          - SPARSE_INVERTED_INDEX on sparse_vector (BM25)

        Args:
            dimension: 语义向量的维度 (如 bge-m3 为1024)。

        Returns:
            True 表示集合已就绪。
        """
        self._dimension = dimension
        client = self._get_client()
        collection_name = settings.MILVUS_COLLECTION

        # 检查集合是否已存在
        has_collection = await self._run_sync(
            client.has_collection, collection_name
        )

        if has_collection:
            logger.info(f"Collection '{collection_name}' 已存在，跳过创建")
            return True

        # ==========================================
        # 创建新集合 (含BM25混合检索Schema)
        # ==========================================
        logger.info(
            f"正在创建 Collection '{collection_name}' "
            f"(维度: {dimension}, 分析器: {settings.MILVUS_ANALYZER})"
        )

        # Step1: 创建 Schema
        schema = client.create_schema(
            auto_id=False,                # 使用自定义UUID
            enable_dynamic_field=True,    # 启用动态字段 (存储metadata)
        )

        # Step2: 添加字段定义
        # --- 主键字段 ---
        schema.add_field(
            field_name=self.FIELD_ID,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )

        # --- 文本字段 (启用分词器, 供BM25使用) ---
        # enable_analyzer=True 让 Milvus 对中文/英文进行分词
        # analyzer_params 指定分词器类型 (chinese=jieba分词)
        schema.add_field(
            field_name=self.FIELD_TEXT,
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": settings.MILVUS_ANALYZER},
        )

        # --- 语义向量字段 (来自 Ollama 嵌入) ---
        schema.add_field(
            field_name=self.FIELD_DENSE,
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )

        # --- BM25 稀疏向量字段 (由 Milvus 内置 BM25 函数自动生成) ---
        # SPARSE_FLOAT_VECTOR 是 Milvus 2.4+ 新增的类型
        # 该字段由 BM25 Function 自动填充, 用户插入时无需提供
        schema.add_field(
            field_name=self.FIELD_SPARSE,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # Step3: 添加 BM25 函数 (文本 → 稀疏向量)
        # BM25 Function 自动对 text 字段分词并计算BM25权重,
        # 结果存入 sparse_vector 字段
        bm25_function = Function(
            name=self.BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=[self.FIELD_TEXT],
            output_field_names=self.FIELD_SPARSE,
        )
        schema.add_function(bm25_function)

        # Step4: 创建索引
        index_params = client.prepare_index_params()

        # --- 语义向量索引: HNSW (高性能图索引) ---
        index_params.add_index(
            field_name=self.FIELD_DENSE,
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 16,                 # 节点最大连接数
                "efConstruction": 200,   # 构建时搜索宽度
            },
        )

        # --- BM25 稀疏向量索引: SPARSE_INVERTED_INDEX ---
        # 基于倒排索引的稀疏向量检索, 专门为 BM25 设计
        index_params.add_index(
            field_name=self.FIELD_SPARSE,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # Step5: 创建 Collection
        await self._run_sync(
            client.create_collection,
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(f"✅ Collection '{collection_name}' 创建成功 (含BM25混合检索)")
        return True

    async def drop_collection(self) -> bool:
        """删除 Collection (谨慎使用)"""
        client = self._get_client()
        await self._run_sync(
            client.drop_collection, settings.MILVUS_COLLECTION
        )
        logger.info(f"Collection '{settings.MILVUS_COLLECTION}' 已删除")
        return True

    # ==================== 数据操作 ====================

    async def insert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        批量插入向量和文本到 Milvus。

        Args:
            vectors: 嵌入向量列表，每个是 float 列表。
            texts: 对应的原始文本列表。
            metadatas: 对应的元数据列表 (可选)。

        Returns:
            插入的chunk ID列表。

        Raises:
            ValueError: 输入参数长度不一致。
        """
        n = len(vectors)
        if len(texts) != n:
            raise ValueError(f"vectors数量({n})与texts数量({len(texts)})不匹配")
        if metadatas and len(metadatas) != n:
            raise ValueError(f"vectors数量({n})与metadatas数量({len(metadatas)})不匹配")

        # 构建插入数据: MilvusClient.insert() 接受 List[Dict] 格式
        chunk_ids = []
        data_rows = []
        for i in range(n):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            row = {
                self.FIELD_ID: chunk_id,
                self.FIELD_TEXT: texts[i],
                self.FIELD_DENSE: vectors[i],   # 仅需提供语义向量, sparse_vector由BM25函数自动生成
            }

            # 将metadata作为动态字段存入
            if metadatas and metadatas[i]:
                row.update(metadatas[i])

            data_rows.append(row)

        # 使用信号量控制并发
        async with self._semaphore:
            client = self._get_client()
            result = await self._run_sync(
                client.insert,
                collection_name=settings.MILVUS_COLLECTION,
                data=data_rows,
            )

        insert_count = result.get("insert_count", 0)
        logger.info(f"成功插入 {insert_count} 条数据到 Milvus")
        return chunk_ids

    # ==================== 检索操作 ====================

    async def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        [向后兼容] 纯语义向量搜索。

        直接委托给 dense_search()。如需混合检索(BM25+向量),
        请使用 hybrid_search()。
        """
        return await self.dense_search(query_vector, top_k, filter_expr)

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        纯语义向量搜索 (Dense Vector Search)。

        使用 HNSW 索引在 dense_vector 字段上进行 COSINE 相似度搜索。

        Args:
            query_vector: 查询向量 (由Ollama嵌入模型生成)。
            top_k: 返回的结果数量。
            filter_expr: Milvus 标量过滤表达式。

        Returns:
            搜索结果列表，每项包含 chunk_id, text, score。
        """
        k = top_k or settings.TOP_K

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(64, k * 8)},  # ef 随 top_k 动态调整
        }

        async with self._semaphore:
            client = self._get_client()
            search_kwargs = {
                "collection_name": settings.MILVUS_COLLECTION,
                "data": [query_vector],
                "limit": k,
                "anns_field": self.FIELD_DENSE,
                "search_params": search_params,
                "output_fields": [self.FIELD_ID, self.FIELD_TEXT],
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr

            results = await self._run_sync(client.search, **search_kwargs)

        return self._format_search_results(results)

    async def sparse_search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 关键词稀疏检索 (Sparse Vector Search)。

        利用 Milvus 内置 BM25 函数自动将查询文本转换为稀疏向量，
        在 sparse_vector 字段上进行 BM25 相似度搜索。

        BM25 的优势:
          - 精确关键词匹配 (适合专有名词、编号、代码等)
          - 词汇级别的重要性加权 (IDF)
          - 与语义检索互补

        Args:
            query_text: 原始查询文本 (Milvus自动分词并转换为BM25稀疏向量)。
            top_k: 返回的结果数量。
            filter_expr: Milvus 标量过滤表达式。

        Returns:
            搜索结果列表，每项包含 chunk_id, text, score。
        """
        k = top_k or settings.TOP_K

        search_params = {
            "metric_type": "BM25",
        }

        async with self._semaphore:
            client = self._get_client()
            search_kwargs = {
                "collection_name": settings.MILVUS_COLLECTION,
                "data": [query_text],     # Milvus BM25 内部自动将文本转为稀疏向量
                "limit": k,
                "anns_field": self.FIELD_SPARSE,
                "search_params": search_params,
                "output_fields": [self.FIELD_ID, self.FIELD_TEXT],
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr

            results = await self._run_sync(client.search, **search_kwargs)

        return self._format_search_results(results)

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
        merge_strategy: str = "rrf",  # "rrf" | "weighted"
    ) -> List[Dict[str, Any]]:
        """
        混合检索: 语义向量 + BM25 关键词 (Hybrid Search)。

        这是推荐的默认检索方式，结合了两种检索范式的优势:
          - 语义检索 (Dense): 理解同义词、语义相似性
          - BM25检索 (Sparse): 精准关键字匹配

        融合策略:
          - "rrf": Reciprocal Rank Fusion (推荐)
            对两组结果的排名进行融合，不需要归一化分数。
            RRF_k=60 是学术界和实践中的经验最优值。
          - "weighted": 加权分数融合
            使用 config 中的 HYBRID_DENSE_WEIGHT 进行线性加权。

        Args:
            query_text: 原始查询文本 (用于BM25)。
            query_vector: 查询语义向量 (用于Dense检索)。
            top_k: 返回的结果数量。
            filter_expr: Milvus 标量过滤表达式。
            merge_strategy: 融合策略 ("rrf" 或 "weighted")。

        Returns:
            融合排序后的搜索结果列表。
        """
        k = top_k or settings.TOP_K

        # ---- 构建语义向量搜索请求 ----
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=self.FIELD_DENSE,
            param={
                "metric_type": "COSINE",
                "params": {"ef": max(64, k * 8)},
            },
            limit=k * 2,  # 多取一些候选, 供融合时筛选
        )

        # ---- 构建 BM25 搜索请求 ----
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=self.FIELD_SPARSE,
            param={"metric_type": "BM25"},
            limit=k * 2,
        )

        # ---- 选择融合排序器 ----
        if merge_strategy == "weighted":
            # 加权融合: 可调节语义 vs 关键词的权重
            dense_weight = settings.HYBRID_DENSE_WEIGHT
            sparse_weight = 1.0 - dense_weight
            ranker = WeightedRanker(dense_weight, sparse_weight)
            logger.debug(
                f"混合检索(加权): dense_weight={dense_weight}, sparse_weight={sparse_weight}"
            )
        else:
            # RRF融合 (默认): 基于排名融合, 无需分数归一化
            ranker = RRFRanker(k=settings.HYBRID_RRF_K)
            logger.debug(f"混合检索(RRF): k={settings.HYBRID_RRF_K}")

        # ---- 执行混合检索 ----
        async with self._semaphore:
            client = self._get_client()
            search_kwargs = {
                "collection_name": settings.MILVUS_COLLECTION,
                "reqs": [dense_req, sparse_req],
                "ranker": ranker,
                "limit": k,
                "output_fields": [self.FIELD_ID, self.FIELD_TEXT],
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr

            results = await self._run_sync(client.hybrid_search, **search_kwargs)

        return self._format_search_results(results)

    # ==================== 结果格式化 ====================

    def _format_search_results(self, results: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        将 Milvus 原始搜索结果格式化为统一结构。

        MilvusClient.search() / hybrid_search() 返回 List[List[Dict]]:
          外层 list → 每个查询向量的结果
          内层 list → 每个命中结果, 格式: {id, distance, entity: {...}}

        Args:
            results: Milvus 原始搜索结果。

        Returns:
            统一格式的搜索结果列表。
        """
        if not results or not results[0]:
            return []

        formatted = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            formatted.append({
                "chunk_id": entity.get(self.FIELD_ID, ""),
                "text": entity.get(self.FIELD_TEXT, ""),
                "score": hit.get("distance", 0.0),
            })

        # 过滤低于阈值的低相关结果
        formatted = [
            r for r in formatted
            if r["score"] >= settings.SIMILARITY_THRESHOLD
        ]

        return formatted

    # ==================== 删除操作 ====================

    async def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """
        根据ID列表删除文档块。

        Args:
            chunk_ids: 要删除的chunk ID列表。

        Returns:
            实际删除的数量。
        """
        if not chunk_ids:
            return 0

        # 构建过滤表达式: id in ["id1", "id2", ...]
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        filter_expr = f'{self.FIELD_ID} in [{ids_str}]'

        async with self._semaphore:
            client = self._get_client()
            result = await self._run_sync(
                client.delete,
                collection_name=settings.MILVUS_COLLECTION,
                filter=filter_expr,
            )

        delete_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info(f"从 Milvus 删除了 {delete_count} 条数据")
        return delete_count

    async def count(self) -> int:
        """
        获取集合中的文档块总数。

        使用 PyMilvus 3.0 的 get_collection_stats() 方法获取行数。

        Returns:
            文档块数量。集合不存在时返回0。
        """
        client = self._get_client()
        try:
            has = await self._run_sync(
                client.has_collection, settings.MILVUS_COLLECTION
            )
            if not has:
                return 0

            # PyMilvus 3.0: get_collection_stats() 返回 {"row_count": N}
            stats = await self._run_sync(
                client.get_collection_stats,
                collection_name=settings.MILVUS_COLLECTION,
            )
            return stats.get("row_count", 0)

        except Exception as e:
            logger.warning(f"获取文档计数失败: {e}")
            return 0

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        获取 Collection 详细信息。

        Returns:
            包含集合名称、状态、文档数量等信息的字典。
        """
        client = self._get_client()
        try:
            has = await self._run_sync(
                client.has_collection, settings.MILVUS_COLLECTION
            )
            doc_count = await self.count()

            return {
                "collection_name": settings.MILVUS_COLLECTION,
                "exists": has,
                "document_count": doc_count,
            }
        except Exception as e:
            logger.error(f"获取Collection信息失败: {e}")
            return {
                "collection_name": settings.MILVUS_COLLECTION,
                "exists": False,
                "document_count": 0,
            }

    async def close(self):
        """
        关闭 Milvus 连接。
        MilvusClient 使用连接池，通常不需要手动关闭，
        但提供此方法用于优雅退出。
        """
        if self._client is not None:
            # MilvusClient 在3.0中通过 close() 释放资源
            try:
                await self._run_sync(self._client.close)
                logger.info("Milvus 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭Milvus连接时出错: {e}")
            finally:
                self._client = None
