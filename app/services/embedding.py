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
from typing import List, Optional
from langfuse import observe

import httpx
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, dimension: Optional[int]  = None, load_type: str = "ollama"):
        """初始化嵌入服务，创建HTTP连接池和并发信号量"""

        self._client: httpx.AsyncClient | None = None
        self._model: SentenceTransformer | None = None

        if load_type == "ollama":
            # httpx 连接池配置: 支持100+并发连接的复用
            # 连接池大小 = 最大并发嵌入请求数 * 2 (留有余量)
            pool_size = min(settings.OLLAMA_EMBED_MAX_CONCURRENT * 2, 200)
            
            self._pool_size = pool_size

            # 并发控制信号量: 限制同时发往Ollama的嵌入请求数
            self._semaphore = asyncio.Semaphore(settings.OLLAMA_EMBED_MAX_CONCURRENT)
        elif load_type == "transformer":
            
            # 并发控制：限制同时进入推理的协程数，防止 GPU OOM
            self._semaphore = asyncio.Semaphore(settings.EMBEDDING_MAX_WORKER)
            self._executor = ThreadPoolExecutor(
                max_workers=settings.EMBEDDING_MAX_WORKER,
                thread_name_prefix="embed"
            )

        # 嵌入向量维度缓存 (首次调用时自动探测)
        self._dimension: int | None = dimension
        self.load_type = load_type

    async def startup(self, load_type: str = "ollama"):

        if load_type == "ollama":
            await self._startup_ollama()
        else:
            await self._startup_transformer()

    async def _startup_transformer(self):
        """加载本地 sentence-transformers 模型并 warm-up"""
        loop = asyncio.get_running_loop()
        # 将同步加载操作放入自定义线程池，避免阻塞事件循环
        self._model = await loop.run_in_executor(
            self._executor,
            self._load_transformer_model
        )
        logger.info("SentenceTransformer 模型加载完成")

        # warm-up，同时探测维度
        vectors = await self.embed_texts(["warm up"])
        if self._dimension is None:
            self._dimension = len(vectors[0])
            logger.info(f"检测到嵌入向量维度: {self._dimension}")

    @staticmethod
    def _load_transformer_model():
        """同步加载模型，供 asyncio.to_thread 调用"""
        model_name = settings.EMBEDDING_TRANSFORMER_MODEL
        return SentenceTransformer(model_name)

    async def _startup_ollama(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(
                    max_connections=self._pool_size,
                    max_keepalive_connections=self._pool_size // 2
                )
            )
            logger.info("嵌入服务连接成功！")

        logger.info("嵌入服务已连接")
        text_embed = await self.embed_texts(["warm up"])
        if self._dimension is None:
            self._dimension = len(text_embed[0])
            logger.info(f"检测到嵌入向量维度: {self._dimension}")

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
            if self.load_type == "ollama":
                return await self._embed_ollama(texts)
            elif self.load_type == "transformer":
                return await self._embed_transformer(texts)
            else:
                raise RuntimeError(f"未知的加载类型: {self.load_type}")

    @observe(name="embed_query", as_type="embedding")
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

    async def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """
        实际执行嵌入请求的核心方法。
        包含重试逻辑和错误处理。
        """
        if self._client is None:
            raise RuntimeError("Embeddings Service未初始化，请先调用startup方法初始化")
        client = self._client
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

    async def _embed_transformer(self, texts: List[str]) -> List[List[float]]:
        """使用本地 SentenceTransformer 模型执行嵌入（在自定义线程池中运行）"""
        if self._model is None:
            raise RuntimeError("SentenceTransformer 模型未加载，请先调用 startup()")

        loop = asyncio.get_running_loop()
        # 将同步编码操作放入自定义线程池，避免占用 asyncio 默认线程池
        embeddings = await loop.run_in_executor(
            self._executor,
            self._model.encode,
            texts
        )
        return embeddings.tolist()

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
        """释放后端资源"""
        if self.load_type == "ollama":
            if self._client is not None:
                await self._client.aclose()
                self._client = None
                logger.info("Ollama HTTP 客户端已关闭")

        elif self.load_type == "transformer":
            if self._executor is not None:
                # 等待所有已提交任务完成，然后关闭线程池
                self._executor.shutdown(wait=True)
                self._executor = None
                logger.info("Transformer 线程池已关闭")
            self._model = None
