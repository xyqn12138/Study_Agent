"""RAGAS evaluation runner for the Study Agent RAG system."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from datasets import Dataset
from ragas import aevaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings

from agent.utils.logger_handler import get_logger

logger = get_logger()

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Regex to extract chunk IDs from tool output (same as server.py)
_CHUNK_ID_RE = re.compile(r"([\w./ -]+?)(L[234]_\d+)")


def _create_ragas_llm():
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        timeout=180,
    )
    return llm_factory("deepseek-chat", client=client, max_tokens=8192)


def _create_ragas_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4"),
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


async def _run_single_query(graph, question: str) -> tuple[str, list[str]]:
    """Run a single question through the LangGraph agent and return (answer, contexts)."""
    input_msg = {"messages": [{"role": "user", "content": question}]}
    config = {"recursion_limit": 50}

    answer = ""
    contexts = []

    async for event in graph.astream_events(input_msg, config=config, version="v2"):
        kind = event.get("event", "")

        if kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            if hasattr(output, "content"):
                output = output.content or ""
            if not isinstance(output, str):
                output = str(output)
            # Extract context text from tool output (each [i] block is a context)
            if output and "[" in output:
                # Split by numbered entries [1], [2], etc.
                blocks = re.split(r"\n\[\d+\]", output)
                for block in blocks:
                    text = block.strip()
                    if text and len(text) > 20:
                        contexts.append(text)

        elif kind == "on_chat_model_end":
            msg = event.get("data", {}).get("output", {})
            if hasattr(msg, "content"):
                answer = msg.content or ""

    # Fallback: get answer from the last AI message in the graph state
    if not answer:
        try:
            final_state = await graph.ainvoke(input_msg, config=config)
            msgs = final_state.get("messages", [])
            for m in reversed(msgs):
                if hasattr(m, "content") and m.content and hasattr(m, "type") and m.type == "ai":
                    answer = m.content
                    break
        except Exception as e:
            logger.warning(f"[Eval] Fallback answer extraction failed: {e}")

    return answer, contexts


class EvalRunner:
    """Runs RAGAS evaluation using the actual LangGraph agent pipeline."""

    def __init__(self):
        from agent.server import _get_graph
        self.graph = _get_graph()

    async def run_evaluation(
        self,
        test_cases: list[dict],
        on_progress: Callable[[int, int, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        total = len(test_cases)
        questions, answers, contexts_list, ground_truths = [], [], [], []

        for i, case in enumerate(test_cases):
            q = case["question"]
            gt = case["ground_truth"]

            if on_progress:
                await on_progress(i, total, q)

            # Run through the actual RAG pipeline
            answer, contexts = await _run_single_query(self.graph, q)

            if not contexts:
                logger.warning(f"[Eval] No contexts retrieved for: {q[:60]}")
                contexts = ["（未检索到相关内容）"]

            questions.append(q)
            answers.append(answer or "（未生成回答）")
            contexts_list.append(contexts)
            ground_truths.append(gt)

        # Debug: log first case
        if questions:
            logger.info(f"[Eval] Sample Q: {questions[0][:80]}")
            logger.info(f"[Eval] Sample A: {answers[0][:120]}")
            logger.info(f"[Eval] Sample C ({len(contexts_list[0])}): {[c[:60] for c in contexts_list[0]]}")
            logger.info(f"[Eval] Sample GT: {ground_truths[0][:80]}")

        # Build RAGAS dataset
        ds = Dataset.from_dict({
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts_list,
            "reference": ground_truths,
        })

        # Run RAGAS evaluation
        ragas_llm = _create_ragas_llm()
        ragas_embed = _create_ragas_embeddings()
        metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()]
        result = await aevaluate(ds, metrics=metrics, llm=ragas_llm, embeddings=ragas_embed, raise_exceptions=False)

        # Extract per-sample scores and compute averages (NaN-safe)
        def safe_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                return round(f, 4) if f == f else None  # NaN check: f != f
            except (TypeError, ValueError):
                return None

        scores = {}
        for name in METRIC_NAMES:
            vals = [safe_float(v) for v in result[name]]
            valid_vals = [v for v in vals if v is not None]
            scores[name] = round(sum(valid_vals) / len(valid_vals), 4) if valid_vals else None

        # Build per-sample details
        details = []
        for i in range(total):
            raw_scores = {name: safe_float(result[name][i]) for name in METRIC_NAMES}
            detail = {
                "question": questions[i],
                "answer": answers[i],
                "ground_truth": ground_truths[i],
                "contexts": contexts_list[i],
                "scores": raw_scores,
            }
            details.append(detail)

        return {"scores": scores, "details": details}
