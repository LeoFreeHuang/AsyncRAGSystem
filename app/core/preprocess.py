"""
================================================================================
文档预处理与清洗模块 (Document Preprocessing & Cleaning)
================================================================================
对加载后的文档进行多阶段清洗和规范化处理，提升检索质量：

生产级清洗流程:
  1. 空白字符规范化 —— 合并多余换行/空格
  2. HTML 标签去除 —— 清理残留的 HTML 标记
  3. 特殊字符清理 —— 移除零宽字符、控制字符等
  4. 敏感信息脱敏 —— 邮箱/手机号等（可选）
  5. 文档分块 —— 按语义边界智能切分
  6. 元数据增强 —— 补充字符统计、语言检测等信息
"""

import regex as re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """文档清洗统计信息"""
    original_chars: int = 0
    cleaned_chars: int = 0
    chunks_before: int = 0
    chunks_after: int = 0
    removed_docs: int = 0  # 被完全清除的文档片段数


class DocumentPreprocessor:
    """
    ============================================================================
    文档预处理器
    提供完整的文档清洗→分块→过滤流水线
    ============================================================================
    """

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
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_cleaning: Optional[bool] = None,
        enable_pii_removal: bool = False,
    ):
        """
        Args:
            chunk_size:       分块大小（字符数）
            chunk_overlap:    分块重叠字符数
            enable_cleaning:  是否启用清洗
            enable_pii_removal: 是否移除敏感信息（PII）
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.enable_cleaning = (
            enable_cleaning
            if enable_cleaning is not None
            else settings.CLEAN_ENABLED
        )
        self.enable_pii_removal = enable_pii_removal
        self.stats = CleaningStats()

        # --- 文本分割器：按语义边界递归切分 ---
        # 优先级: 段落(\n\n) > 换行(\n) > 句号(。) > 空格 > 字符
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # 公共接口：完整清洗流水线
    # ------------------------------------------------------------------

    def process(self, documents: List[Document]) -> Tuple[List[Document], CleaningStats]:
        """
        执行完整的文档清洗流水线:
          清洗 → 分块 → 过滤空文档

        Args:
            documents: 原始文档列表

        Returns:
            (cleaned_documents, stats): 清洗后的文档列表和统计信息
        """
        self.stats = CleaningStats()
        self.stats.chunks_before = len(documents)
        self.stats.original_chars = sum(len(d.page_content) for d in documents)

        # --- 阶段 1: 文本清洗 (Text Cleaning) ---
        if self.enable_cleaning:
            documents = [self._clean_single(doc) for doc in documents]
            # 过滤掉清洗后为空的文档
            documents = [d for d in documents if d.page_content.strip()]

        # --- 阶段 2: 文档分块 (Chunking) ---
        #documents = self._chunk_documents(documents)

        documents = self._chunk_documents_grouped(documents)

        # --- 阶段 3: 后过滤 (Post-filtering) ---
        documents = self._filter_documents(documents)

        # --- 统计 ---
        self.stats.chunks_after = len(documents)
        self.stats.cleaned_chars = sum(len(d.page_content) for d in documents)

        logger.info(
            f"文档预处理完成: {self.stats.chunks_before} → {self.stats.chunks_after} 片段"
            f" ({self.stats.original_chars} → {self.stats.cleaned_chars} 字符)"
        )
        return documents, self.stats

    # ------------------------------------------------------------------
    # 清洗方法
    # ------------------------------------------------------------------

    def _clean_single(self, doc: Document) -> Document:
        """
        对单个文档执行清洗操作

        清洗步骤:
          1) 统一换行符 (Windows \r\n → \n)
          2) 移除 HTML/XML 标签
          3) 合并多个连续空白行
          4) 移除零宽字符和控制字符
          5) 规范化中英文标点空格
          6) 敏感信息脱敏（可选）
        """
        text = doc.page_content

        # Step 1: 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: 移除 HTML/XML 标签
        text = re.sub(r"<[^>]+>", " ", text)
        # 移除 HTML 实体
        html_entities = {
            "&nbsp;": " ", "&lt;": "<", "&gt;": ">",
            "&amp;": "&", "&quot;": '"', "&#x27;": "'",
            "&ldquo;": '"', "&rdquo;": '"', "&mdash;": "—",
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)

        # Step 3: 合并多个空白行（保留段落分隔的单个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 合并行内多余空格
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Step 4: 移除不可见的零宽字符和控制字符
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
        # 保留常见的换行/制表，移除其他控制字符（ASCII 0-31, 排除 \n \t）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Step 5: 中英文混排规范化
        # 中文与英文/数字之间添加空格
        text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z0-9])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fff])", r"\1 \2", text)

        # Step 6: 敏感信息脱敏（可选）
        if self.enable_pii_removal:
            text = self._remove_pii(text)

        # 去除首尾空白
        text = text.strip()

        # 更新文档
        cleaned_doc = Document(
            page_content=text,
            metadata={**doc.metadata, "cleaned": True}
        )
        return cleaned_doc

    def _remove_pii(self, text: str) -> str:
        """
        移除 / 脱敏个人身份信息 (PII)

        包括:
          - 邮箱地址 → [EMAIL]
          - 手机号（中国） → [PHONE]
          - 身份证号 → [ID]
          - IPv4 地址 → [IP]
        """
        # 邮箱
        text = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[EMAIL]", text
        )
        # 中国手机号
        text = re.sub(r"1[3-9]\d{9}", "[PHONE]", text)
        # 中国身份证号 (18位)
        text = re.sub(r"\d{17}[\dXx]", "[ID]", text)
        # IPv4
        text = re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "[IP]", text
        )
        return text

    # ------------------------------------------------------------------
    # 分块方法
    # ------------------------------------------------------------------

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行智能分块

        策略:
          - 对普通文本使用 RecursiveCharacterTextSplitter 按语义边界切分
          - 保留原始元数据并附加 chunk 序号
        """
        all_chunks: List[Document] = []

        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                # 内容已在 chunk_size 以内，无需切分
                doc.metadata["chunk_index"] = 0
                all_chunks.append(doc)
            else:
                # 切分并附加 chunk index
                sub_docs = self.text_splitter.split_documents([doc])
                for i, sub_doc in enumerate(sub_docs):
                    sub_doc.metadata["chunk_index"] = i
                    sub_doc.metadata["chunk_total"] = len(sub_docs)
                all_chunks.extend(sub_docs)

        return all_chunks

    def _chunk_documents_grouped(self, documents: List[Document]) -> List[Document]:
        """
        按源文件分组后分块，确保同一文件的 chunk_index 全局连续。
        保留每个切片原始所在页面的 metadata。
        """
        # 按 source 字段分组（确保每个文档都有 source 元数据）
        groups = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            groups[source].append(doc)

        all_chunks = []
        for source, docs in groups.items():
            # 按页码排序，保证切片顺序正确（如果加载器提供了 page 字段）
            docs.sort(key=lambda d: d.metadata.get("page", 0))
            
            current_idx = 0  # 该文件的全局切片计数器
            for doc in docs:
                if len(doc.page_content) <= self.chunk_size:
                    # 内容较短，无需切分，直接作为一个 chunk
                    doc.metadata["chunk_index"] = current_idx
                    all_chunks.append(doc)
                    current_idx += 1
                else:
                    # 需要切分
                    sub_docs = self.text_splitter.split_documents([doc])
                    for sub_doc in sub_docs:
                        sub_doc.metadata["chunk_index"] = current_idx
                        # 保留原始 metadata（如 page 等），split_documents 会自动继承
                        current_idx += 1
                    all_chunks.extend(sub_docs)
            
            # 可选：记录该文档总切片数
            for chunk in all_chunks[-current_idx:]:
                chunk.metadata["chunk_total"] = current_idx

        return all_chunks

    # ------------------------------------------------------------------
    # 过滤方法
    # ------------------------------------------------------------------

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        过滤低质量文档片段

        过滤规则:
          1) 空白或仅含标点符号的片段
          2) 字符数过少的片段（< 20 字符）
          3) 重复率过高的片段（与同一文档内的其他片段比对）
        """
        filtered: List[Document] = []
        seen_hashes: set = set()

        for doc in documents:
            content = doc.page_content.strip()

            # 规则 1 & 2: 空内容或过短
            if not content or len(content) < 20:
                self.stats.removed_docs += 1
                continue

            # 仅含标点和空白
            if re.match(r"^[\s\p{P}]+$", content):
                self.stats.removed_docs += 1
                continue

            # 规则 3: 简单去重（基于内容哈希）
            content_hash = hash(content)
            if content_hash in seen_hashes:
                self.stats.removed_docs += 1
                continue
            seen_hashes.add(content_hash)

            filtered.append(doc)

        return filtered
