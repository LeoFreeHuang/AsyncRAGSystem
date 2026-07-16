"""
============================================================
AsyncRAGSystem - 文本分块策略模块
提供多种文本分割器，用于将长文档切分为适合嵌入的语义块
============================================================
"""

import re
from typing import List, Tuple
from app.config import settings


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器。

    策略: 按优先级依次尝试用不同分隔符切分文本，
    优先保持段落/句子完整性，逐级降级到字符级切分。
    分隔符优先级: 段落(\\n\\n) > 句子(。！？.!?) > 短语(，,;；) > 空格 > 字符

    使用示例:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = splitter.split_text("很长的文档文本...")
    """

    # 分隔符优先级列表 (从粗粒度到细粒度)
    _SEPARATORS: List[str] = [
        "\n\n",     # 段落分隔 (Markdown/纯文本)
        "\n",       # 换行
        "。",       # 中文句号
        "！",       # 中文感叹号
        "？",       # 中文问号
        "！",       # 中文感叹号
        ".",        # 英文句号
        "!",        # 英文感叹号
        "?",        # 英文问号
        "；",       # 中文分号
        ";",        # 英文分号
        "，",       # 中文逗号
        ",",        # 英文逗号
        " ",        # 空格
        "",         # 最终降级: 逐字符切分
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        """
        Args:
            chunk_size: 每个文本块的最大字符数。默认使用全局配置。
            chunk_overlap: 相邻块之间的重叠字符数。默认使用全局配置。
            separators: 自定义分隔符优先级列表。
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self._separators = separators or self._SEPARATORS

        # 参数校验
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )

    def split_text(self, text: str) -> List[str]:
        """
        将长文本递归分割为指定大小的文本块。

        Args:
            text: 待分割的原始文本。

        Returns:
            文本块列表，每个块长度不超过 chunk_size。
        """
        # 去除首尾空白，规范化换行
        text = text.strip()
        if not text:
            return []

        # 如果文本本身不超过 chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]

        # 递归分割
        chunks = self._split_recursive(text)
        return chunks

    def _split_recursive(self, text: str, sep_idx: int = 0) -> List[str]:
        """
        递归分割核心逻辑。

        Args:
            text: 当前待分割文本。
            sep_idx: 当前使用的分隔符在 _SEPARATORS 中的索引。

        Returns:
            分割后的文本块列表。
        """
        # 递归终止: 文本已足够短
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # 所有分隔符都已尝试，使用强制字符级切分
        if sep_idx >= len(self._separators):
            return self._force_split(text)

        separator = self._separators[sep_idx]

        # 空字符串分隔符表示逐字符切分
        if separator == "":
            return self._force_split(text)

        # 使用当前分隔符切分
        splits = text.split(separator)

        # 如果分隔符无效 (整个文本没有被切分)，尝试下一级分隔符
        if len(splits) == 1:
            return self._split_recursive(text, sep_idx + 1)

        # 合并短片段，构建最终chunks
        return self._merge_splits(splits, separator, sep_idx)

    def _merge_splits(
        self, splits: List[str], separator: str, sep_idx: int
    ) -> List[str]:
        """
        合并被切分的片段，确保每个chunk不超过chunk_size，
        同时尽可能保持语义完整性。

        对于超过chunk_size的单个片段，递归使用下一级分隔符继续切分。
        """
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len: int = 0

        for split in splits:
            split_len = len(split)

            # 情况1: 单个片段超过chunk_size → 递归用更细粒度分隔符
            if split_len > self.chunk_size:
                # 先保存当前积累的chunk
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # 递归切分超长片段
                sub_chunks = self._split_recursive(split, sep_idx + 1)
                chunks.extend(sub_chunks)
                continue

            # 情况2: 加入当前片段后超过chunk_size → 开启新chunk
            # 计算加入后长度 (考虑是否需要分隔符)
            sep_len = len(separator) if current_chunk else 0
            if current_len + sep_len + split_len > self.chunk_size:
                chunks.append(separator.join(current_chunk))
                # 重叠策略: 保留上一个chunk的尾部内容作为上下文
                current_chunk = self._get_overlap_tail(current_chunk)
                # 重新计算 current_len (overlap后可能为空或更短)
                current_len = sum(len(s) for s in current_chunk) + len(separator) * max(0, len(current_chunk) - 1)

            # 情况3: 正常追加到当前chunk
            # 注意: added_len 在情况2中可能因 current_chunk 被重置而变化, 需重新计算
            sep_len = len(separator) if current_chunk else 0
            current_chunk.append(split)
            current_len += sep_len + split_len

        # 处理最后一个chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _get_overlap_tail(self, chunks: List[str]) -> List[str]:
        """
        从当前chunk列表中提取尾部内容作为重叠上下文。
        策略: 从后往前取片段，直到累计长度达到 chunk_overlap。
        """
        if not chunks or self.chunk_overlap <= 0:
            return []

        overlap_chunks: List[str] = []
        accumulated = 0
        for chunk in reversed(chunks):
            if accumulated >= self.chunk_overlap:
                break
            overlap_chunks.insert(0, chunk)
            accumulated += len(chunk)

        return overlap_chunks

    def _force_split(self, text: str) -> List[str]:
        """
        强制字符级切分 (最后手段)。
        当所有语义分隔符都无效时使用。
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    def split_documents(
        self, documents: List[dict]
    ) -> List[dict]:
        """
        批量分割多个文档，为每个文本块保留源文档的元数据。

        Args:
            documents: 文档列表，每个文档为 {"text": str, "metadata": dict}。

        Returns:
            文本块列表，每个块包含 text, metadata, chunk_index 字段。
        """
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.split_text(text)
            for idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": idx},
                })
        return all_chunks
