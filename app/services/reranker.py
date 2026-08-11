"""
============================================================
AsyncRAGSystem - 重排服务 (Reranker Service)
负责检索文档的重排
============================================================
"""

import asyncio
from typing import List, Union, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
import torch
import logging

from app.config import settings

logger = logging.getLogger(__name__)

class RerankService:
    """
    基于 SentenceTransformers CrossEncoder 的异步精排器。
    适合已通过检索获得 top-k 候选文档，需进行精细语义重排序的场景。
    """
    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,   # 也可用 "cross-encoder/ms-marco-MiniLM-L-6-v2" 等
        use_fp16: bool = settings.RERANKER_USER_FP16,                       # 显存紧张时 GPU 上可开启，CPU 推理建议关闭
        max_length: int = settings.RERANKER_MAX_LENGTH,
        batch_size: int = settings.RERANKER_BATCH_SIZE,
        device: str = settings.RERANKER_DEVICE                           # 推荐用 cpu，完全不影响 LLM 生成
    ):
        # 模型实例化在 __init__ 中同步执行（启动时调用，不阻塞异步循环）
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = True if use_fp16 and self.device.startswith("cuda") else False
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device or self.device,
        )
        if use_fp16 and self.device.startswith("cuda"):
            self.model.half()  # 将底层 transformer 转为半精度
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        异步精排方法。

        参数:
            query: 用户查询字符串
            candidates: 待重排的文档文本列表（初检 top-k）
            return_scores: True 则返回 (原始索引, 分数, 文本) 列表；False 则只返回排序后的文本列表

        返回:
            按相关性分数从高到低排列的结果
        """
        if not candidates:
            return []

        # 构造 CrossEncoder 标准输入格式：[(query, doc1), (query, doc2), ...]
        pairs = [(query, doc.get("text", "")) for doc in candidates]

        # 将同步推理封装为异步，在线程池执行，避免事件循环阻塞
        scores = await asyncio.to_thread(
            self._predict, pairs
        )

        # 将分数与索引、文本打包并排序
        indexed = [
            (idx, float(score), doc)
            for idx, (doc, score) in enumerate(zip(candidates, scores))
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for _, score, doc in indexed:
            doc["rerank_score"] = score
            results.append(doc)
            
        return results[:settings.TOP_K]

    def _predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """同步推理方法，在线程池中调用。"""
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        # 保证返回 list，即使只有一对
        if isinstance(scores, float):
            scores = [scores]
        return list(scores)