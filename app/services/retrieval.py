"""
============================================================
AsyncRAGSystem - RAG检索服务 (Retrieval Service)
编排完整的RAG问答流水线: 缓存检查 → 嵌入 → 混合检索 → 增强 → 生成
============================================================

设计要点:
1. 端到端的RAG流水线编排
2. Redis两级缓存 (精确+语义) → 命中则跳过整个流水线
3. BM25 + 语义向量 混合检索 → 兼顾关键词匹配和语义理解
4. 检索结果去重与上下文拼接
5. 来源追溯 (返回检索到的源文档)
6. 支持流式和非流式两种生成模式
7. 完整的耗时统计，方便性能调优

流水线步骤:
  0. 缓存检查 (L1精确 → L2语义) → 命中直接返回
  1. Query Embedding: 将用户问题向量化
  2. Hybrid Search: BM25关键词检索 + Dense语义检索 → RRF融合
  3. Context Assembly: 将检索结果拼装为LLM上下文
  4. Generation: 调用LLM生成增强回答
  5. 缓存写入: 将结果写入Redis (两级缓存)
"""

import logging
import time
from typing import AsyncGenerator, List, Optional
import asyncio

from app.config import settings
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStoreService
from app.services.cache import CacheService
from app.api.schemas import SourceDocument

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    RAG 检索增强生成服务。

    编排完整的 "缓存检查 → 嵌入 → 混合检索 → 增强 → 生成 → 缓存写入" 流水线。

    使用示例:
        rag = RetrievalService(embedding, llm, vector_store, cache)
        response = await rag.query("什么是向量数据库?")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: VectorStoreService,
        cache_service: Optional[CacheService] = None,
    ):
        self._embedding = embedding_service
        self._llm = llm_service
        self._vector_store = vector_store
        self._cache = cache_service

    async def _get_cached_or_none(self, question: str) -> Optional[dict]:
        """缓存检查，返回缓存数据或 None"""
        if self._cache:
            return await self._cache.get(question)
        return None

    async def _set_cache(self, question: str, response: dict) -> None:
        """写入缓存（如果 cache 存在）"""
        if self._cache:
            await self._cache.set(question, response)

    async def _retrieve_and_assemble(
        self, question: str, top_k: int
    ) -> tuple[str, List[SourceDocument], str, float, float]:
        """
        执行嵌入 + 混合检索 + 上下文组装。
        返回: (context, sources, search_method, embed_time_ms, search_time_ms)
        """

        # ================================================
        # Step 1: 嵌入查询问题 (生成Dense向量)
        # ================================================
        embed_start = time.monotonic()
        query_vector = await self._embedding.embed_query(question)
        embed_time = (time.monotonic() - embed_start) * 1000

        # ================================================
        # Step 2: 混合检索 (BM25 + 语义向量)
        # 使用 Milvus 内置 hybrid_search + RRF 融合
        # ================================================
        search_start = time.monotonic()
        try:
            search_results = await self._vector_store.hybrid_search(
                query_text=question,
                query_vector=query_vector,
                top_k=top_k,
                merge_strategy="rrf",
            )
            search_method = "hybrid(RRF)"
        except Exception as e:
            logger.warning(f"混合检索失败, 降级为纯语义检索: {e}")
            search_results = await self._vector_store.dense_search(
                query_vector=query_vector, top_k=top_k
            )
            search_method = "dense(fallback)"

        search_time = (time.monotonic() - search_start) * 1000
        logger.debug(f"{search_method}检索耗时: {search_time:.1f}ms, 结果数: {len(search_results)}")

        # 构建源文档列表（转为 Pydantic 模型）
        sources = [
            SourceDocument(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                score=r.get("score", 0.0),
            )
            for r in search_results
        ]

        # ================================================
        # Step 3: 构建上下文 (拼接检索到的文档片段)
        # ================================================
        context = self._assemble_context(search_results)
        return context, sources, search_method, embed_time, search_time

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        执行RAG问答 (非流式)。

        完整流水线:
          0. 检查Redis缓存 (命中则直接返回)
          1. 查询嵌入 → 混合检索 (BM25 + Dense)
          2. 上下文构建 → LLM生成
          3. 结果缓存 (供后续重复查询复用)

        Args:
            question: 用户问题。
            top_k: 检索文档数量。
            temperature: LLM生成温度。

        Returns:
            {"answer": str, "sources": List[SourceDocument], "processing_time_ms": float,
             "cached": bool, "cache_stats": dict}
        """
        start_time = time.monotonic()

        # ================================================
        # Step 0: 缓存检查 (L1精确 → L2语义)
        # ================================================
        if self._cache:
            cached = await self._cache.get(question)
            if cached is not None:
                total_time = (time.monotonic() - start_time) * 1000
                logger.info(
                    f"🎯 缓存命中! 非流式返回(跳过整个RAG流水线)"
                )
                cached["processing_time_ms"] = round(total_time, 1)
                cached["cached"] = True
                return cached

        k = top_k or settings.TOP_K

        # ---- 检索与上下文 ----
        context, sources, search_method, embed_time, search_time = (
            await self._retrieve_and_assemble(question, k)
        )

        # ================================================
        # Step 4: LLM生成回答
        # ================================================
        gen_start = time.monotonic()
        answer = await self._llm.generate(
            question=question,
            context=context,
            temperature=temperature,
        )
        gen_time = (time.monotonic() - gen_start) * 1000
        logger.debug(f"LLM生成耗时: {gen_time:.1f}ms")

        total_time = (time.monotonic() - start_time) * 1000
        logger.info(
            f"RAG查询完成: 总耗时={total_time:.0f}ms "
            f"(嵌入={embed_time:.0f}ms, {search_method}={search_time:.0f}ms, 生成={gen_time:.0f}ms)"
        )

        response = {
            "answer": answer,
            "sources": sources,
            "processing_time_ms": round(total_time, 1),
            "model": settings.LLM_MODEL,
            "cached": False,
            "search_method": search_method,
        }

        # ================================================
        # Step 5: 写入缓存
        # ================================================
        await self._set_cache(question, response)
        return response

    async def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        执行RAG问答 (流式SSE)。

        检索阶段完成后，将LLM的生成结果逐token yield。

        注意: 流式模式下不进行缓存 (无法缓存生成器)。

        Args:
            question: 用户问题。
            top_k: 检索文档数量。
            temperature: LLM生成温度。

        Yields:
            LLM生成的文本token。
        """

        start_time = time.monotonic()

        if self._cache:
            cached = await self._cache.get(question)
            if cached is not None:
                logger.info("🎯 缓存命中! 模拟流式返回(跳过整个RAG流水线)")
                answer = cached.get("answer", "")
                if not isinstance(answer, str):
                    answer = str(answer)  # 防御性转换
                # 按字数分组，添加延迟
                chunk_size = 2  # 可调整
                for i in range(0, len(answer), chunk_size):
                    yield answer[i:i+chunk_size]
                    await asyncio.sleep(0.025)  # 25ms，可调
                return

        k = top_k or settings.TOP_K

        # ---- 检索与上下文 ----
        context, sources, search_method, embed_time, search_time = (
            await self._retrieve_and_assemble(question, k)
        )
       
        # Step 4: 流式生成
        tokens = []
        gen_start = time.monotonic()
        async for token in self._llm.generate_stream(
            question=question,
            context=context,
            temperature=temperature,
        ):
            tokens.append(token)
            yield token

        gen_time = (time.monotonic() - gen_start) * 1000
        logger.debug(f"LLM生成耗时: {gen_time:.1f}ms")

        total_time = (time.monotonic() - start_time) * 1000
        logger.info(
            f"RAG查询完成: 总耗时={total_time:.0f}ms "
            f"(嵌入={embed_time:.0f}ms, {search_method}={search_time:.0f}ms, 生成={gen_time:.0f}ms)"
        )

        response = {
            "answer": "".join(tokens),
            "sources": sources,
            "processing_time_ms": round(total_time, 1),
            "model": settings.LLM_MODEL,
            "cached": False,
            "search_method": search_method,
        }

        await self._set_cache(question, response)

    # qwen3.5 上下文窗口约 32768 tokens, 保守估计每token≈2字符, 预留50%给prompt模板+回答
    _MAX_CONTEXT_CHARS = 30000  # 约15000 tokens的安全上限

    def _assemble_context(self, search_results: List[dict]) -> str:
        """
        将检索结果拼接为LLM可用的上下文字符串。

        策略:
        - 按相似度降序排列
        - 每个片段标注序号和相关度
        - 截断过长上下文，防止超出LLM上下文窗口 (qwen3.5: 32768 tokens)

        Args:
            search_results: 检索结果列表。

        Returns:
            格式化的上下文字符串 (不超过 _MAX_CONTEXT_CHARS 字符)。
        """
        if not search_results:
            return ""

        # 过滤低相关度结果
        filtered = [
            r for r in search_results
            if r.get("score", 0) >= settings.SIMILARITY_THRESHOLD
        ]

        # 去重 (基于文本内容的简单去重)
        seen_texts = set()
        unique_results = []
        for r in filtered:
            text = r.get("text", "").strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(r)

        if not unique_results:
            return ""

        # 拼接为编号列表, 同时监控总长度 (避免超出LLM上下文窗口)
        context_parts = []
        total_chars = 0
        for idx, result in enumerate(unique_results, 1):
            text = result.get("text", "").strip()
            score = result.get("score", 0)
            part = f"[文档片段 {idx}] (相关度: {score:.2f})\n{text}"
            part_len = len(part)

            # 超出上下文窗口上限时截断
            if total_chars + part_len > self._MAX_CONTEXT_CHARS:
                remaining = self._MAX_CONTEXT_CHARS - total_chars
                if remaining > 100:  # 至少保留有意义的片段
                    part = part[:remaining] + "...(截断)"
                    context_parts.append(part)
                logger.warning(
                    f"上下文长度已达上限 ({self._MAX_CONTEXT_CHARS}字符), "
                    f"已截断, 实际使用了 {idx}/{len(unique_results)} 个片段"
                )
                break

            context_parts.append(part)
            total_chars += part_len

        return "\n\n".join(context_parts)
