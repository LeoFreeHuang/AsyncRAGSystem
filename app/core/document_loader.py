"""
================================================================================
多格式文档加载器 (Multi-Format Document Loader)
================================================================================
支持加载 PDF、HTML、Excel(.xlsx/.xls)、Word(.docx/.doc)、TXT 等格式的文档。
使用 LangChain 社区加载器进行统一的文档加载，返回标准化的 Document 对象列表。

生产级特性:
  - 支持超大文件分页加载，避免 OOM
  - 文件类型自动识别与路由
  - 加载进度回调
  - 异常隔离（单个文件失败不影响批量加载）
"""

import hashlib
import logging
from typing import List, Optional, Callable, Dict
from pathlib import Path
import asyncio

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
)

from app.config import settings
from app.core.preprocess import DocumentPreprocessor

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    ============================================================================
    通用文档加载器
    根据文件扩展名自动选择合适的 LangChain 加载器，将文档解析为
    langchain_core.documents.Document 列表。
    ============================================================================
    """

    # 文件扩展名 → 加载器类映射表
    LOADER_MAP: Dict[str, type] = {
        ".pdf": PyPDFLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """
        初始化文档加载器

        Args:
            chunk_size:  文本分块大小，默认使用全局配置
            chunk_overlap: 分块重叠字符数
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.supported_exts = settings.SUPPORTED_EXTENTIONS
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        self.doc_preprocessor = DocumentPreprocessor()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件，返回 Document 列表

        Args:
            file_path: 文件的绝对路径或相对路径

        Returns:
            List[Document]: 解析后的文档对象列表

        Raises:
            ValueError: 文件格式不支持或文件过大
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path).resolve()

        # --- 文件存在性校验 ---
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # --- 文件大小校验 ---
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(
                f"文件过大: {self._format_size(file_size)} > "
                f"{self._format_size(self.max_file_size)}"
            )

        # --- 扩展名校验 ---
        ext = path.suffix.lower()
        if ext not in self.supported_exts:
            raise ValueError(
                f"不支持的文件格式: {ext}，支持: {self.supported_exts}"
            )

        logger.info(f"开始加载文档: {path.name} ({self._format_size(file_size)})")

        # --- 获取对应的加载器类 ---
        loader_cls = self.LOADER_MAP.get(ext)
        if loader_cls is None:
            raise ValueError(f"未注册的加载器: {ext}")

        try:
            # --- 实例化加载器并加载 ---
            if ext == ".txt":
                loader = loader_cls(str(path), encoding="utf-8")
            else:
                loader = loader_cls(str(path))
            documents = await loader.aload() #或者采用await asyncio.to_thread(loader.load)

            # --- 为每个 Document 注入元数据 ---
            for doc in documents:
                doc.metadata.update({
                    "source_file": path.name,
                    "source_path": str(path),
                    "file_type": ext,
                    "file_size": file_size,
                    "doc_id": path.name,
                    "char_count": len(doc.page_content),
                })

            logger.info(f"文档加载完成: {path.name}, 共 {len(documents)} 个片段")
            return documents

        except Exception as e:
            logger.error(f"文档加载失败: {path.name}, 错误: {e}")
            raise RuntimeError(f"文档加载失败 [{path.name}]: {e}") from e

    async def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        on_error: str = "skip",  # "skip" | "raise" | "log"
    ) -> List[Document]:
        """
        批量加载目录下所有支持的文档

        Args:
            dir_path:           目录路径
            recursive:          是否递归加载子目录
            progress_callback:  进度回调 (current, total, filename)
            on_error:           错误处理策略
                - "skip": 跳过失败文件，继续处理
                - "raise": 遇到错误立即抛出
                - "log":  记录日志后继续

        Returns:
            List[Document]: 所有文档片段的列表
        """
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"不是有效目录: {path}")

        # --- 收集所有支持的文件 ---
        all_files: List[Path] = []
        if recursive:
            for ext in self.supported_exts:
                all_files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in self.supported_exts:
                all_files.extend(path.glob(f"*{ext}"))

        # 按文件名排序，保证处理顺序稳定
        all_files = sorted(all_files, key=lambda f: f.name)

        total = len(all_files)
        logger.info(f"目录扫描完成: {dir_path}, 找到 {total} 个文档")

        all_documents: List[Document] = []
        for idx, file_path in enumerate(all_files, start=1):
            try:
                docs = await self.load_file(str(file_path))
                all_documents.extend(docs)

                if progress_callback:
                    progress_callback(idx, total, file_path.name)

            except Exception as e:
                if on_error == "raise":
                    raise
                elif on_error == "skip":
                    logger.warning(f"跳过文件: {file_path.name}, 原因: {e}")
                else:  # "log"
                    logger.error(f"加载失败(继续): {file_path.name}, 错误: {e}")

        logger.info(f"目录加载完成: {len(all_documents)} 个文档片段")
        return all_documents

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_doc_id(doc: Document) -> str:
        """
        为文档片段生成唯一 ID
        基于 source + page 组合确保唯一性
        """
        raw = (
            doc.metadata.get("source", "")
            + str(doc.metadata.get("page", ""))
            + doc.page_content[:100]  # 取前200字符做哈希，避免过长
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """将字节数格式化为可读字符串"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes //= 1024
        return f"{size_bytes:.1f} TB"
