"""
============================================================
AsyncRAGSystem - RAGAS评估服务 (RAGAS Service)
负责评估系统质量
============================================================
"""

import os
import sys
import types

# ============================================================
# 补丁：解决 ragas 0.4.3 导入 langchain_community.chat_models.vertexai 报错
# ============================================================
try:
    from langchain_google_vertexai import ChatVertexAI  # type: ignore
except ImportError:
    print("请先运行: pip install langchain-google-vertexai")
    sys.exit(1)

if "langchain_community.chat_models.vertexai" not in sys.modules:
    fake_module = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_module.ChatVertexAI = ChatVertexAI  # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = fake_module
# ============================================================

from langfuse import get_client
import json
import pandas as pd
from pandas import DataFrame
import asyncio

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextUtilization
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas import experiment

from openai import AsyncOpenAI

# 获取当前文件的父目录，也就是项目根目录 (AsyncRAGSystem)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将根目录插入到 sys.path 的最前面
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.config import settings


langfuse = get_client()

def fetch_lanfuse_dataset() -> DataFrame:

    data = []

    #拉取最近100个traces
    trace_response = langfuse.api.trace.list(limit=100)
    print(f"拉取trace个数：{len(trace_response.data)}")

    if trace_response:
        trace_data = trace_response.data
        for trace in trace_data:
            trace_id = trace.id
            question = ""
            full_answer = ""
            contexts = []

            if isinstance(trace.input, dict):
                question = (
                    trace.input.get("kwargs", {})
                    .get("query_input", {})
                    .get("question", "")
                ) or ""

            if isinstance(trace.output, str):
                lines = [line.strip() for line in trace.output.split("\n") if line.strip()]

                for line in lines[:-1]:
                    if line.startswith("data: "):
                        json_str = line.replace("data: ", "")
                        try:
                            token = json.loads(json_str).get("token", "")
                            full_answer += token
                        except json.JSONDecodeError as e:
                            print(f"JSON解析报错：{e}")
                            continue

            full_trace = langfuse.api.trace.get(trace_id=trace_id)
            for observation in (full_trace.observations or []):
                if settings.RERANKER_ENABLED and observation.name == "rerank":
                    output = observation.output
                    for content in output:
                        contexts.append(content.get("text", "").replace("\n", ""))
                elif observation.name == "hybrid_search":
                    output = observation.output
                    for content in output:
                        contexts.append(content.get("text", "").replace("\n", ""))

            data.append({
                "question": question,
                "answer": full_answer,
                "contexts": contexts,
            })

    for d in data:
        print(d)
        print("-" * 50)
            
    df = pd.DataFrame(data)
    print(df)
    return df
            

def evalute_rag(df: DataFrame) -> DataFrame:

    # ============================================================
    # 使用 Ollama 的 OpenAI 兼容 API
    # ============================================================
    async_ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    # 使用 ragas 0.4 官方 llm_factory 创建 LLM
    # 注意: ragas 默认 max_tokens=1024, 结构化JSON输出很容易被截断,
    # 报错 IncompleteOutputException: The output is incomplete due to a max_tokens length limit.
    evaluator_llm = llm_factory(
        "qwen3.5:9b",
        provider="openai",
        client=async_ollama_client,
        max_tokens=16384*2,
    )

    # 使用 ragas 0.4 原生 OpenAIEmbeddings 连接 Ollama
    evaluator_embeddings = OpenAIEmbeddings(
        client=async_ollama_client,
        model="bge-m3:latest",
    )

    # 实例化指标
    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    contextUtilization = ContextUtilization(llm=evaluator_llm)

    


    # ============================================================
    # 不使用 evaluate()，直接用 ascore() 逐条评分
    # evaluate() 在 0.4.3 中不兼容 metrics.collections 新式指标
    # ============================================================
    scores_faithfulness = []
    scores_answer_relevancy = []
    scores_context_precision = []

    # Ollama 单卡串行推理, 限制并发数, 避免请求堆积导致超时
    sem = asyncio.Semaphore(max(1, settings.OLLAMA_LLM_MAX_CONCURRENT))

    async def limited(coro):
        async with sem:
            return await coro

    async def score_row(row):
        """对单条数据并行计算所有指标, 单条失败不影响整体"""
        try:
            f_score, ar_score, cp_score = await asyncio.gather(
                limited(faithfulness.ascore(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["contexts"],
                )),
                limited(answer_relevancy.ascore(
                    user_input=row["question"],
                    response=row["answer"],
                )),
                limited(contextUtilization.ascore(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["contexts"],
                )),
            )
            return f_score.value, ar_score.value, cp_score.value
        except Exception as e:
            q = str(row.get("question", ""))[:50]
            print(f"[评分失败-跳过] question={q}... 原因: {type(e).__name__}: {e}")
            return None, None, None

    async def run_all():
        tasks = [score_row(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        for f, ar, cp in results:
            scores_faithfulness.append(f)
            scores_answer_relevancy.append(ar)
            scores_context_precision.append(cp)

    asyncio.run(run_all())

    # 将评分结果写回 DataFrame
    df["faithfulness"] = scores_faithfulness
    df["answer_relevancy"] = scores_answer_relevancy
    df["context_precision"] = scores_context_precision

    score_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    failed_count = int(df[score_cols].isna().any(axis=1).sum())
    print(f"\n成功评分: {len(df) - failed_count}/{len(df)} 条, 失败(跳过): {failed_count} 条")

    print(df[["question"] + score_cols].head())
    print(f"\n平均值 (忽略失败项):")
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            print(f"  {col}: {vals.mean():.4f}")
        else:
            print(f"  {col}: 无有效评分")

    return df

if __name__ == "__main__":
    df = fetch_lanfuse_dataset()
    evaluate_result = evalute_rag(df)