"""
============================================================
AsyncRAGSystem - Redis 缓存服务 (Cache Service)
提供两级缓存: 精确关键字缓存 + 语义归一化缓存
============================================================

设计要点:
1. 两级缓存策略:
   - L1 精确缓存: 对完全相同的查询直接返回缓存结果 (MD5 key)
   - L2 语义缓存: 对归一化后的查询文本匹配 (忽略大小写/空格/标点)
2. 使用 redis.asyncio 异步客户端, 不阻塞事件循环
3. 可配置的 TTL (过期时间), 防止缓存无限增长
4. 缓存穿透保护: 空结果也缓存 (短TTL)
5. 优雅降级: Redis 不可用时自动跳过缓存, 不影响主流程

缓存Key设计:
    精确缓存: rag:exact:{md5(原始问题)}
    语义缓存: rag:semantic:{md5(归一化问题)}

为什么这样设计:
    - 精确缓存命中率最高, 适合高频重复查询 (如热门问题)
    - 语义缓存覆盖同义改写场景 (如 "什么是RAG" vs "啥是RAG")
    - 不采用向量相似度缓存, 避免额外嵌入开销 (且需Redis Stack)
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Redis 两级缓存服务。

    L1 - 精确关键字缓存:
        对原始 query 做 MD5, 缓存完整响应。
        适用场景: 完全相同的重复查询。

    L2 - 语义归一化缓存:
        对 query 归一化 (小写/去空格/去标点) 后做 MD5, 缓存完整响应。
        适用场景: 同义改写查询 ("什么是RAG?" ≈ "什么是rag")。

    使用示例:
        cache = CacheService()
        await cache.connect()

        # 写入缓存
        await cache.set_exact("什么是RAG?", {"answer": "...", "sources": [...]})

        # 读取缓存 (先L1后L2)
        cached = await cache.get("什么是RAG?")
        if cached:
            return cached  # 缓存命中, 跳过整个RAG流水线
    """

    # Redis Key 前缀
    KEY_PREFIX_EXACT = "rag:exact"
    KEY_PREFIX_SEMANTIC = "rag:semantic"

    def __init__(self):
        """初始化缓存服务 (不建立连接, 由 connect() 显式连接)"""
        self._redis: Optional[aioredis.Redis] = None
        self._enabled = settings.CACHE_ENABLED

        # 缓存统计 (用于监控)
        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0

    # ==================== 连接管理 ====================

    async def connect(self):
        """
        建立 Redis 异步连接。

        连接失败时自动降级: 设置为 disabled 状态,
        后续所有 get/set 操作静默跳过。
        """
        if not self._enabled:
            logger.info("Redis 缓存已禁用 (CACHE_ENABLED=false)")
            return

        try:
            self._redis = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,       # 自动解码为 str
                socket_connect_timeout=5,     # 连接超时5秒
                socket_keepalive=True,        # TCP keepalive
                health_check_interval=30,     # 健康检查间隔
            )
            # 验证连接
            await self._redis.ping()
            logger.info(f"✅ Redis 缓存已连接: {settings.redis_url}")

        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败, 缓存已降级禁用: {e}")
            self._enabled = False
            self._redis = None

    async def disconnect(self):
        """关闭 Redis 连接, 释放资源"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis 缓存连接已关闭")

    @property
    def enabled(self) -> bool:
        """缓存是否可用"""
        return self._enabled and self._redis is not None

    # ==================== 缓存读取 ====================

    async def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        两级缓存查询 (L1→L2 级联)。

        先查精确缓存, 未命中则查语义缓存。

        Args:
            question: 用户原始问题文本。

        Returns:
            缓存命中时返回完整响应 dict, 未命中返回 None。
        """
        if not self.enabled:
            return None

        # L1: 精确关键字缓存
        result = await self._get_exact(question)
        if result is not None:
            self._hits_exact += 1
            logger.debug(f"🎯 L1精确缓存命中: {question[:50]}...")
            return result

        # L2: 语义归一化缓存
        result = await self._get_semantic(question)
        if result is not None:
            self._hits_semantic += 1
            logger.debug(f"🔍 L2语义缓存命中: {question[:50]}...")
            return result

        self._misses += 1
        return None

    async def _get_exact(self, question: str) -> Optional[Dict[str, Any]]:
        """读取精确关键字缓存"""
        key = self._make_key(self.KEY_PREFIX_EXACT, question)
        return await self._redis_get_json(key)

    async def _get_semantic(self, question: str) -> Optional[Dict[str, Any]]:
        """读取语义归一化缓存"""
        normalized = self._normalize(question)
        key = self._make_key(self.KEY_PREFIX_SEMANTIC, normalized)
        return await self._redis_get_json(key)

    async def _redis_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """从 Redis 读取 JSON 并反序列化"""
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Redis 读取失败 key={key}: {e}")
            return None

    # ==================== 缓存写入 ====================

    async def set(self, question: str, response: Dict[str, Any]):
        """
        同时写入两级缓存。

        精确缓存 TTL 较长 (重复查询价值高),
        语义缓存 TTL 较短 (归一化可能误匹配)。

        Args:
            question: 用户原始问题。
            response: 完整的 RAG 响应 (含 answer, sources 等)。
        """
        if not self.enabled:
            return

        await self._set_exact(question, response)
        await self._set_semantic(question, response)

    async def _set_exact(self, question: str, response: Dict[str, Any]):
        """写入精确关键字缓存"""
        key = self._make_key(self.KEY_PREFIX_EXACT, question)
        await self._redis_set_json(key, response, settings.CACHE_EXACT_TTL)

    async def _set_semantic(self, question: str, response: Dict[str, Any]):
        """写入语义归一化缓存"""
        normalized = self._normalize(question)
        key = self._make_key(self.KEY_PREFIX_SEMANTIC, normalized)

        # 同一归一化文本可能有多个原始问题映射, 使用更短TTL降低误匹配风险
        await self._redis_set_json(key, response, settings.CACHE_SEMANTIC_TTL)

    async def _redis_set_json(self, key: str, value: Dict[str, Any], ttl: int):
        """将 dict 序列化为 JSON 写入 Redis, 并设置过期时间"""
        try:
            # 对 SourceDocument 等 Pydantic 对象做序列化处理
            serialized = self._serialize(value)
            await self._redis.setex(key, ttl, json.dumps(serialized, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Redis 写入失败 key={key}: {e}")

    # ==================== 缓存清除 ====================

    async def invalidate(self, question: Optional[str] = None):
        """
        清除缓存。

        Args:
            question: 若指定, 仅清除与该问题相关的缓存;
                      若为 None, 清除所有 RAG 缓存 (flush by prefix)。
        """
        if not self.enabled:
            return

        if question:
            # 精确清除
            exact_key = self._make_key(self.KEY_PREFIX_EXACT, question)
            semantic_key = self._make_key(
                self.KEY_PREFIX_SEMANTIC, self._normalize(question)
            )
            await self._redis.delete(exact_key, semantic_key)
            logger.debug(f"已清除问题缓存: {question[:50]}...")
        else:
            # 全量清除 (使用SCAN避免阻塞)
            await self._flush_by_prefix(f"{self.KEY_PREFIX_EXACT}:*")
            await self._flush_by_prefix(f"{self.KEY_PREFIX_SEMANTIC}:*")
            logger.info("已清除所有RAG缓存")

    async def _flush_by_prefix(self, pattern: str):
        """按前缀模式删除所有匹配的key (使用SCAN, 避免KEYS阻塞)"""
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor, match=pattern, count=100
            )
            if keys:
                await self._redis.delete(*keys)
            if cursor == 0:
                break

    # ==================== 统计信息 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存命中率统计"""
        total = self._hits_exact + self._hits_semantic + self._misses
        return {
            "enabled": self._enabled and self._redis is not None,
            "hits_exact": self._hits_exact,
            "hits_semantic": self._hits_semantic,
            "misses": self._misses,
            "hit_rate": (
                round((self._hits_exact + self._hits_semantic) / total, 3)
                if total > 0 else 0.0
            ),
        }

    # ==================== 工具方法 ====================

    def _make_key(self, prefix: str, text: str) -> str:
        """
        生成 Redis Key: {prefix}:{md5(text)}

        使用 MD5 确保 key 长度可控 (避免原始问题过长),
        同时 MD5 碰撞概率在实际场景中可忽略不计。
        """
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def _normalize(self, text: str) -> str:
        """
        文本归一化处理。

        处理步骤:
        1. 转小写
        2. 移除标点符号
        3. 合并连续空白字符
        4. 去除首尾空白

        示例:
            "什么是 RAG 系统？？" → "什么是 rag 系统"
            "What's RAG?" → "whats rag"
        """
        text = text.lower().strip()
        # 移除中英文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        # 合并空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _serialize(self, obj: Any) -> Any:
        """
        递归序列化对象为 JSON 兼容格式。
        处理 Pydantic 模型、datetime 等特殊类型。
        """
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        elif hasattr(obj, "model_dump"):  # Pydantic v2
            return self._serialize(obj.model_dump())
        elif hasattr(obj, "dict"):  # Pydantic v1
            return self._serialize(obj.dict())
        elif hasattr(obj, "isoformat"):  # datetime/date
            return obj.isoformat()
        else:
            return obj
