"""
============================================================
AsyncRAGSystem - 嵌入服务 (Embedding Service)
通过 Ollama API 调用本地嵌入模型，将文本转换为向量表示
============================================================

设计要点:
1. 使用 httpx.AsyncClient 连接池实现高并发异步请求
2. 支持批量嵌入 (一次请求处理多条文本，提升吞吐)
3. 使用 asyncio.Semaphore 控制并发，避免压垮Ollama
4. 自动重试机制，处理临时性网络故障
"""

import asyncio
import logging
from typing import List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    文本嵌入服务。

    通过 Ollama 的 /api/embed 端点将文本转换为稠密向量。
    支持单条和批量嵌入，自动管理HTTP连接池。

    使用示例:
        service = EmbeddingService()
        vectors = await service.embed_texts(["你好，世界", "RAG系统"])
    """

    def __init__(self):
        """初始化嵌入服务，创建HTTP连接池和并发信号量"""
        # httpx 连接池配置: 支持100+并发连接的复用
        # 连接池大小 = 最大并发嵌入请求数 * 2 (留有余量)
        pool_size = min(settings.OLLAMA_EMBED_MAX_CONCURRENT * 2, 200)
        self._client: httpx.AsyncClient | None = None
        self._pool_size = pool_size

        # 并发控制信号量: 限制同时发往Ollama的嵌入请求数
        self._semaphore = asyncio.Semaphore(settings.OLLAMA_EMBED_MAX_CONCURRENT)

        # 嵌入向量维度缓存 (首次调用时自动探测)
        self._dimension: int | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        延迟初始化 HTTP 客户端 (在事件循环中创建)。
        httpx.AsyncClient 必须在异步上下文中创建。
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),          # 嵌入超时60秒
                limits=httpx.Limits(
                    max_connections=self._pool_size,
                    max_keepalive_connections=self._pool_size // 2,
                ),
            )
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为嵌入向量列表。

        对于多条文本，Ollama支持批量处理，比逐条调用效率高得多。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            嵌入向量列表，每个向量的维度取决于所选模型。
            例如 bge-m3 返回 1024维向量。

        Raises:
            httpx.HTTPError: Ollama API 调用失败。
            ValueError: 返回向量数量与输入不匹配。
        """
        if not texts:
            return []

        # 使用信号量控制并发
        async with self._semaphore:
            return await self._do_embed(texts)

    async def embed_query(self, text: str) -> List[float]:
        """
        嵌入单条查询文本 (便捷方法)。

        Args:
            text: 查询文本。

        Returns:
            单个嵌入向量。
        """
        results = await self.embed_texts([text])
        return results[0]

    async def _do_embed(self, texts: List[str]) -> List[List[float]]:
        """
        实际执行嵌入请求的核心方法。
        包含重试逻辑和错误处理。
        """
        client = await self._get_client()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    settings.ollama_embed_url,
                    json={
                        "model": settings.EMBEDDING_MODEL,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Ollama /api/embed 返回 {"embeddings": [[...], [...], ...]}
                embeddings = data.get("embeddings", [])

                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"嵌入向量数量 ({len(embeddings)}) 与输入文本数量 ({len(texts)}) 不匹配"
                    )

                # 缓存向量维度
                if embeddings and self._dimension is None:
                    self._dimension = len(embeddings[0])
                    logger.info(f"检测到嵌入向量维度: {self._dimension}")

                return embeddings

            except httpx.HTTPStatusError as e:
                logger.warning(f"嵌入请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避

            except httpx.RequestError as e:
                logger.warning(f"网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        # 理论上不会到达这里，但保持类型安全
        raise RuntimeError("嵌入请求失败: 已达最大重试次数")

    @property
    async def dimension(self) -> int:
        """
        获取嵌入向量的维度。
        如果尚未探测，发送一条测试文本获取维度。
        """
        if self._dimension is None:
            vectors = await self.embed_texts(["维度探测文本"])
            self._dimension = len(vectors[0])
        return self._dimension

    async def close(self):
        """关闭HTTP客户端，释放连接资源"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("EmbeddingService HTTP客户端已关闭")
