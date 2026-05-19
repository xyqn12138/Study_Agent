from typing import List, Dict, Any, Optional
from agent.models.embedding_model import EmbeddingModel
from agent.rag.milvus_manage import MilvusManage
from agent.utils.logger_handler import get_logger
from pymilvus import AnnSearchRequest, RRFRanker
import os

logger = get_logger()

SHORT_QUERY_THRESHOLD = int(os.getenv("RETRIEVE_SHORT_QUERY_THRESHOLD", "15"))
DEFAULT_HYBRID_LIMIT = int(os.getenv("RETRIEVE_HYBRID_LIMIT", "15"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

_HYDE_PROMPT = (
    "请针对以下问题，从教材的角度写一段简短的知识点总结（不超过200字），"
    "直接输出内容，不要加标题和前缀：\n\n{query}"
)

_REWRITE_PROMPT = (
    "你是一个搜索专家。请将以下用户查询改写为更适合在文档库中进行语义检索的表达方式，"
    "保持原意，但增加可能的同义关键词。直接输出改写后的查询：\n\n查询：{query}"
)


class Retriever:
    def __init__(self, model_name: str = "dashscope", dimensions: int = 1024):
        self.embedding_model = EmbeddingModel(model_name=model_name, dimensions=dimensions)
        self.milvus_manager = MilvusManage()
        self._chat_model = None
        self._reranker = None
        self.collection_name = os.getenv('COLLECTION_NAME')

    @property
    def chat_model(self):
        if self._chat_model is None:
            from agent.models.chat_model import chat_model as _cm
            self._chat_model = _cm
        return self._chat_model

    @property
    def reranker(self):
        if self._reranker is None:
            from agent.models.reranker_model import RerankerModel
            self._reranker = RerankerModel()
        return self._reranker

    @staticmethod
    def _is_short_query(query: str) -> bool:
        return len(query.strip()) <= SHORT_QUERY_THRESHOLD

    def generate_hyde(self, query: str) -> str:
        prompt = _HYDE_PROMPT.format(query=query)
        response = self.chat_model.invoke(prompt)
        hyde_text = response.content.strip()
        logger.info(f"HyDE generated for query: {query} -> {hyde_text[:80]}...")
        return hyde_text

    def rewrite_query(self, query: str) -> str:
        prompt = _REWRITE_PROMPT.format(query=query)
        response = self.chat_model.invoke(prompt)
        rewritten_query = response.content.strip()
        logger.info(f"Original query: {query} -> Rewritten: {rewritten_query}")
        return rewritten_query

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        search_levels: Optional[List[int]] = None,
        scope_filter: Optional[str] = None,
    ) -> List[Dict]:
        search_levels = search_levels or [3, 4]
        client = self.milvus_manager._get_connect()

        dense_vec = self.embedding_model.embed_queries(query)[0]

        search_filter = f"chunk_level in {search_levels}"
        if scope_filter:
            search_filter = f"({search_filter}) && ({scope_filter})"

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="text_dense",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=search_filter
        )

        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
            expr=search_filter
        )

        res = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["text", "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "filename", "title_path", "content_type", "page_number", "image_paths"]
        )

        results = []
        if res and len(res) > 0:
            for hit in res[0]:
                results.append({
                    "text": hit.get("text"),
                    "chunk_id": hit.get("chunk_id"),
                    "parent_chunk_id": hit.get("parent_chunk_id"),
                    "root_chunk_id": hit.get("root_chunk_id"),
                    "chunk_level": hit.get("chunk_level"),
                    "filename": hit.get("filename"),
                    "title_path": hit.get("title_path", ""),
                    "content_type": hit.get("content_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "image_paths": hit.get("image_paths", ""),
                    "score": hit.score
                })

        return results

    def _build_scope_filter(self, parent_ids: List[str], field: str = "parent_chunk_id") -> Optional[str]:
        if not parent_ids:
            return None
        escaped = ", ".join(f'"{cid}"' for cid in parent_ids)
        return f"{field} in [{escaped}]"

    def rerank(self, query: str, results: List[Dict], top_n: int = 3) -> List[Dict]:
        if not results:
            return []
        documents = [hit["text"] for hit in results]
        logger.info(f"Reranking {len(results)} results with qwen3-rerank, top_n={top_n}")
        rerank_results = self.reranker.rerank(query=query, documents=documents, top_n=top_n)
        reranked: List[Dict] = []
        for rr in rerank_results:
            original = results[rr.index]
            original["rerank_score"] = rr.relevance_score
            reranked.append(original)
        return reranked

    def fetch_multi_layer_context(self, search_results: List[Dict]) -> List[Dict]:
        final_contexts = []
        processed_ids = set()

        all_ancestor_ids: set[str] = set()
        for hit in search_results:
            cid = hit["chunk_id"]
            if cid in processed_ids:
                continue
            processed_ids.add(cid)
            if hit.get("root_chunk_id"):
                all_ancestor_ids.add(hit["root_chunk_id"])
            if hit.get("parent_chunk_id"):
                all_ancestor_ids.add(hit["parent_chunk_id"])

        ancestor_map: dict[str, dict] = {}
        if all_ancestor_ids:
            ancestor_list = list(all_ancestor_ids)
            for i in range(0, len(ancestor_list), 100):
                batch = ancestor_list[i : i + 100]
                rows = self.milvus_manager.query_by_chunk_ids(
                    batch, output_fields=["text", "chunk_id", "chunk_level", "title_path", "parent_chunk_id"]
                )
                for row in rows:
                    ancestor_map[row["chunk_id"]] = row

            missing_l2_ids: set[str] = set()
            for hit in search_results:
                if hit["chunk_level"] == 4 and hit.get("parent_chunk_id"):
                    parent = ancestor_map.get(hit["parent_chunk_id"])
                    if parent and parent.get("chunk_level") == 3 and parent.get("parent_chunk_id"):
                        missing_l2_ids.add(parent["parent_chunk_id"])
            missing_l2_ids -= ancestor_map.keys()
            if missing_l2_ids:
                batch = list(missing_l2_ids)
                for i in range(0, len(batch), 100):
                    rows = self.milvus_manager.query_by_chunk_ids(
                        batch[i : i + 100], output_fields=["text", "chunk_id", "chunk_level", "title_path", "parent_chunk_id"]
                    )
                    for row in rows:
                        ancestor_map[row["chunk_id"]] = row

        processed_ids.clear()
        for hit in search_results:
            chunk_id = hit["chunk_id"]
            if chunk_id in processed_ids:
                continue

            context = {
                "chunk_id": hit["chunk_id"],
                "search_hit": hit["text"],
                "level": hit["chunk_level"],
                "filename": hit["filename"],
                "title_path": hit.get("title_path", ""),
                "content_type": hit.get("content_type", ""),
                "page_number": hit.get("page_number", 0),
                "image_paths": hit.get("image_paths", ""),
                "chunk1_text": "",
                "chunk2_text": "",
                "chunk3_text": hit["text"] if hit["chunk_level"] == 3 else "",
                "chunk4_text": hit["text"] if hit["chunk_level"] == 4 else "",
            }

            ids_to_check: set[str] = set()
            if hit.get("root_chunk_id"):
                ids_to_check.add(hit["root_chunk_id"])
            if hit.get("parent_chunk_id"):
                ids_to_check.add(hit["parent_chunk_id"])
            if hit["chunk_level"] == 4 and hit.get("parent_chunk_id"):
                parent_node = ancestor_map.get(hit["parent_chunk_id"])
                if parent_node and parent_node.get("parent_chunk_id"):
                    ids_to_check.add(parent_node["parent_chunk_id"])

            for aid in ids_to_check:
                node = ancestor_map.get(aid)
                if not node:
                    continue
                level = node["chunk_level"]
                if level == 1:
                    context["chunk1_text"] = node["text"]
                elif level == 2:
                    context["chunk2_text"] = node["text"]
                elif level == 3:
                    context["chunk3_text"] = node["text"]

            if hit["chunk_level"] == 3:
                context["chunk3_text"] = hit["text"]
            elif hit["chunk_level"] == 2:
                context["chunk2_text"] = hit["text"]

            final_contexts.append(context)
            processed_ids.add(chunk_id)

        return final_contexts

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        skip_rewrite: bool = False,
        use_hyde: bool = False,
    ) -> List[Dict]:
        logger.info(f"Retrieve: query='{query}', limit={limit}, skip_rewrite={skip_rewrite}, use_hyde={use_hyde}")
        short = self._is_short_query(query)
        hybrid_limit = max(DEFAULT_HYBRID_LIMIT, limit)
        rerank_top_n = min(DEFAULT_RERANK_TOP_N, limit)

        if use_hyde:
            search_query = self.generate_hyde(query)
            search_results = self.hybrid_search(search_query, limit=hybrid_limit, search_levels=[3, 4])
            logger.info("HyDE mode: direct L3/L4 search")

        elif short:
            if not skip_rewrite:
                anchor_query = self.rewrite_query(query)
            else:
                anchor_query = query

            l2_results = self.hybrid_search(anchor_query, limit=3, search_levels=[2])
            logger.info(f"Short query: found {len(l2_results)} L2 anchors")

            l2_ids = [hit["chunk_id"] for hit in l2_results if hit.get("chunk_id")]
            scope_l3 = self._build_scope_filter(l2_ids, field="parent_chunk_id")
            if scope_l3:
                l3_results = self.hybrid_search(anchor_query, limit=hybrid_limit, search_levels=[3], scope_filter=scope_l3)
                logger.info(f"Two-stage: scoped L3 search returned {len(l3_results)} results")

                l3_ids = [hit["chunk_id"] for hit in l3_results if hit.get("chunk_id")]
                scope_l4 = self._build_scope_filter(l3_ids, field="parent_chunk_id")
                if scope_l4:
                    l4_results = self.hybrid_search(anchor_query, limit=hybrid_limit, search_levels=[4], scope_filter=scope_l4)
                else:
                    l4_results = []
                search_results = l3_results + l4_results
                logger.info(f"Two-stage: total L3={len(l3_results)}, L4={len(l4_results)}")
            else:
                search_results = self.hybrid_search(anchor_query, limit=hybrid_limit, search_levels=[3, 4])
                logger.info("Two-stage fallback: no L2 scope found, searching all L3/L4")

        else:
            if not skip_rewrite:
                search_query = self.rewrite_query(query)
            else:
                search_query = query
            search_results = self.hybrid_search(search_query, limit=hybrid_limit, search_levels=[3, 4])
            logger.info("Complex query: direct L3/L4 search")

        reranked_results = self.rerank(query, search_results, top_n=rerank_top_n)

        # --- Pack expansion: enrich results with learned knowledge packs ---
        try:
            from agent.rag.knowledge_pack import get_pack_manager
            pack_mgr = get_pack_manager()
            hit_ids = [r["chunk_id"] for r in reranked_results]
            pack_expansions = pack_mgr.get_packs_for_chunks(hit_ids)
            if pack_expansions:
                existing_ids = set(hit_ids)
                new_ids: list[str] = []
                for related_ids in pack_expansions.values():
                    for cid in related_ids:
                        if cid not in existing_ids:
                            new_ids.append(cid)
                            existing_ids.add(cid)
                # Cap to prevent context pollution
                new_ids = new_ids[:12]
                if new_ids:
                    extra = self.milvus_manager.query_by_chunk_ids(
                        new_ids,
                        output_fields=["text", "chunk_id", "parent_chunk_id", "root_chunk_id",
                                       "chunk_level", "filename", "title_path", "content_type",
                                       "page_number", "image_paths"],
                    )
                    for c in extra:
                        c["_from_pack"] = True
                        reranked_results.append(c)
                    logger.info(f"[Pack] Expanded {len(extra)} chunk(s) from knowledge packs")
        except Exception as e:
            logger.warning(f"[Pack] Expansion error (non-fatal): {e}")

        final_contexts = self.fetch_multi_layer_context(reranked_results)
        return final_contexts


if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("讲讲电路布线例题？")
    for r in results:
        print(r)
        print("-" * 70)
