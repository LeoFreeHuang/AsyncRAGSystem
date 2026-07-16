"""
============================================================
AsyncRAGSystem - 大语言模型服务 (LLM Service)
通过 Ollama API 调用本地部署的大模型进行文本生成
============================================================

设计要点:
1. 使用 httpx.AsyncClient 连接池实现异步调用
2. asyncio.Semaphore 限制并发LLM请求 (单GPU串行推理特性)
3. 支持普通生成和流式生成 (SSE)
4. RAG专用提示词模板，引导模型基于检索上下文回答
5. 自动超时和重试机制
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# RAG 提示词模板
# ============================================================

RAG_SYSTEM_PROMPT = """你是一个专业的知识问答助手。请严格基于以下提供的参考文档来回答用户的问题。

## 回答规则:
1. **只能基于参考文档内容回答**，不要使用你自身的知识。
2. 如果参考文档中没有相关信息，请明确告知用户"根据现有文档无法回答该问题"。
3. 回答要准确、简洁、有条理，使用中文。
4. 如果适用，请在回答中引用具体的文档片段。
5. 当信息来自多个文档片段时，请综合整理后给出连贯的回答。

## 参考文档:
{context}

## 用户问题:
{question}

## 回答:"""


class LLMService:
    """
    大语言模型服务。

    通过 Ollama 的 /api/generate 端点进行文本生成。
    支持基于RAG上下文的增强生成。

    使用示例:
        service = LLMService()
        answer = await service.generate(question="什么是RAG?", context="RAG是检索增强生成...")
        # 流式生成
        async for token in service.generate_stream(question="...", context="..."):
            print(token, end="")
    """

    def __init__(self):
        """初始化LLM服务，创建HTTP连接池"""
        pool_size = min(settings.OLLAMA_LLM_MAX_CONCURRENT * 2, 50)
        self._client: httpx.AsyncClient | None = None
        self._pool_size = pool_size

        # LLM并发控制: Ollama在单GPU上实际只能串行推理
        # 设置上限避免请求堆积超时
        self._semaphore = asyncio.Semaphore(settings.OLLAMA_LLM_MAX_CONCURRENT)

    async def _get_client(self) -> httpx.AsyncClient:
        """延迟初始化HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.LLM_TIMEOUT),
                limits=httpx.Limits(
                    max_connections=self._pool_size,
                    max_keepalive_connections=self._pool_size // 2,
                ),
            )
        return self._client

    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建RAG提示词。

        将检索到的文档上下文和用户问题填入提示词模板。

        Args:
            question: 用户原始问题。
            context: 从向量库检索到的相关文档内容 (已拼接)。

        Returns:
            完整的提示词字符串。
        """
        # 如果上下文为空，使用简化提示词
        if not context or not context.strip():
            return f"请回答以下问题:\n\n{question}"

        return RAG_SYSTEM_PROMPT.format(context=context, question=question)

    async def generate(
        self,
        question: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        同步生成回答 (非流式)。

        等待模型完整生成后一次性返回全部文本。

        Args:
            question: 用户问题。
            context: 检索到的相关文档上下文。
            temperature: 生成温度 (None则使用全局配置)。
            max_tokens: 最大生成token数。

        Returns:
            模型生成的完整回答文本。
        """
        prompt = self._build_prompt(question, context)
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        async with self._semaphore:
            return await self._do_generate(prompt, temp, max_tok)

    async def generate_stream(
        self,
        question: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答 (SSE)。

        每生成一个token立即yield，适合前端打字机效果。

        Args:
            question: 用户问题。
            context: 检索上下文。
            temperature: 生成温度。
            max_tokens: 最大token数。

        Yields:
            逐个生成的文本token。
        """
        prompt = self._build_prompt(question, context)
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        async with self._semaphore:
            async for token in self._do_generate_stream(prompt, temp, max_tok):
                yield token

    async def _do_generate(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """
        实际执行非流式生成请求。
        """
        client = await self._get_client()
        start_time = time.monotonic()

        try:
            response = await client.post(
                settings.ollama_generate_url,
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,          # 非流式模式
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama /api/generate 非流式返回 {"response": "..."}
            answer = data.get("response", "")
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(f"LLM生成完成, 耗时: {elapsed:.0f}ms, 长度: {len(answer)}字符")
            return answer

        except httpx.TimeoutException:
            logger.error(f"LLM请求超时 (>{settings.LLM_TIMEOUT}s)")
            raise RuntimeError(f"LLM生成超时，请尝试缩短问题或减少上下文")

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM请求失败: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"LLM服务异常: {e}")

    async def _do_generate_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """
        实际执行流式生成请求。
        Ollama 流式模式下每行返回一个JSON，包含 "response" 字段。
        """
        client = await self._get_client()
        start_time = time.monotonic()
        total_tokens = 0

        try:
            async with client.stream(
                "POST",
                settings.ollama_generate_url,
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            total_tokens += 1
                            yield token
                        # 检查是否完成
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(f"LLM流式生成完成, 耗时: {elapsed:.0f}ms, tokens: {total_tokens}")

        except httpx.TimeoutException:
            logger.error("LLM流式请求超时")
            yield "\n\n[生成超时，请重试]"

        except Exception as e:
            logger.error(f"LLM流式生成异常: {e}")
            yield f"\n\n[生成出错: {e}]"

    async def close(self):
        """关闭HTTP客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("LLMService HTTP客户端已关闭")
