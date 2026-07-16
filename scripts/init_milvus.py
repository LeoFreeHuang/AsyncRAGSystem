"""
============================================================
AsyncRAGSystem - Milvus 初始化脚本 (v0.2.0)
独立运行此脚本以手动创建/重置 Milvus Collection (含BM25 Schema)

Schema (v0.2.0):
  - id (VARCHAR, PK): UUID主键
  - text (VARCHAR, enable_analyzer): 原始文本 (BM25分词)
  - dense_vector (FLOAT_VECTOR, 1024d): Ollama语义向量
  - sparse_vector (SPARSE_FLOAT_VECTOR): Milvus BM25自动生成
  - 动态字段: 自定义metadata

索引:
  - HNSW on dense_vector (COSINE)
  - SPARSE_INVERTED_INDEX on sparse_vector (BM25)

用法:
    # 创建 Collection (保留已有数据)
    python scripts/init_milvus.py

    # 删除并重建 Collection (清空所有数据, 含BM25 Schema)
    python scripts/init_milvus.py --reset

    # 仅检查连接状态
    python scripts/init_milvus.py --check
"""

import argparse
import asyncio
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.vector_store import VectorStoreService
from app.services.embedding import EmbeddingService


async def check_connection():
    """检查 Milvus 连接状态"""
    print("=" * 50)
    print("检查 Milvus 连接...")
    print(f"URI: {settings.milvus_uri}")

    store = VectorStoreService()
    try:
        info = await store.get_collection_info()
        print(f"✅ Milvus 连接成功!")
        print(f"   Collection: {info['collection_name']}")
        print(f"   存在: {info['exists']}")
        print(f"   文档数: {info['document_count']}")
    except Exception as e:
        print(f"❌ Milvus 连接失败: {e}")
        return False

    await store.close()
    return True


async def init_collection(reset: bool = False):
    """
    初始化 Milvus Collection。

    Args:
        reset: 如果为 True，先删除已有 Collection 再重建。
    """
    print("=" * 50)
    print("AsyncRAGSystem - Milvus 初始化")
    print(f"Collection: {settings.MILVUS_COLLECTION}")
    print(f"嵌入模型: {settings.EMBEDDING_MODEL}")
    print("=" * 50)

    # Step 1: 探测嵌入向量维度
    print("\n[1/3] 探测嵌入向量维度...")
    embed_service = EmbeddingService()
    try:
        test_vectors = await embed_service.embed_texts(["维度探测文本"])
        dimension = len(test_vectors[0])
        print(f"  ✅ 嵌入维度: {dimension}")
    except Exception as e:
        print(f"  ❌ 嵌入探测失败: {e}")
        print("  请确保 Ollama 已启动并已拉取嵌入模型:")
        print(f"    ollama pull {settings.EMBEDDING_MODEL}")
        await embed_service.close()
        return
    await embed_service.close()

    # Step 2: 创建/重置 Collection
    store = VectorStoreService()

    if reset:
        print(f"\n[2/3] 删除已有 Collection '{settings.MILVUS_COLLECTION}'...")
        try:
            await store.drop_collection()
            print("  ✅ 已删除")
        except Exception as e:
            print(f"  ⚠️  删除失败 (可能不存在): {e}")

    print(f"\n[2/3] 创建 Collection '{settings.MILVUS_COLLECTION}' (维度={dimension})...")
    try:
        await store.ensure_collection(dimension)
        print("  ✅ Collection 已就绪")
    except Exception as e:
        print(f"  ❌ 创建失败: {e}")
        await store.close()
        return

    # Step 3: 验证
    print("\n[3/3] 验证 Collection...")
    info = await store.get_collection_info()
    print(f"  Collection: {info['collection_name']}")
    print(f"  状态: {'✅ 正常' if info['exists'] else '❌ 异常'}")
    print(f"  文档数: {info['document_count']}")

    await store.close()

    print("\n" + "=" * 50)
    print("✅ Milvus 初始化完成!")
    print("=" * 50)
    print("\n现在可以启动 RAG 服务:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4")
    print("\n或开发模式:")
    print("  python -m app.main")


async def main():
    parser = argparse.ArgumentParser(
        description="AsyncRAGSystem Milvus 初始化工具"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="删除已有 Collection 并重建 (清空所有数据)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查 Milvus 连接状态，不创建 Collection",
    )
    args = parser.parse_args()

    if args.check:
        success = await check_connection()
        sys.exit(0 if success else 1)
    else:
        await init_collection(reset=args.reset)


if __name__ == "__main__":
    asyncio.run(main())
