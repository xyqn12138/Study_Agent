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
        """Build hierarchical contexts from search results.

        Rules:
        - L3 hits: return L3 text as search_hit, with L1/L2 ancestors.
        - L4 hits: do NOT return L4 text. Instead return the L3 parent text as
          search_hit, with L1/L2 ancestors. Multiple L4 hits sharing the same
          L3 parent are merged into a single entry.
        """
        final_contexts: list[dict] = []

        # --- Phase 1: classify hits ---
        l3_hits: dict[str, dict] = {}   # chunk_id -> hit
        l4_by_parent: dict[str, list[dict]] = {}  # parent_l3_id -> [l4 hits]
        other_hits: list[dict] = []     # L2 or other

        for hit in search_results:
            level = hit["chunk_level"]
            cid = hit["chunk_id"]
            if level == 4 and hit.get("parent_chunk_id"):
                l4_by_parent.setdefault(hit["parent_chunk_id"], []).append(hit)
            elif level == 3:
                if cid not in l3_hits:
                    l3_hits[cid] = hit
            else:
                other_hits.append(hit)

        # L3 parents that are covered by L4 hits — skip them as standalone entries
        l3_ids_from_l4: set[str] = set(l4_by_parent.keys())

        # --- Phase 2: collect all ancestor IDs to fetch ---
        all_ancestor_ids: set[str] = set()
        # From L3 hits (standalone, not covered by L4)
        for hit in l3_hits.values():
            if hit["chunk_id"] in l3_ids_from_l4:
                continue
            if hit.get("root_chunk_id"):
                all_ancestor_ids.add(hit["root_chunk_id"])
            if hit.get("parent_chunk_id"):
                all_ancestor_ids.add(hit["parent_chunk_id"])
        # From L4 hits (need L3 parent + its ancestors)
        for parent_id, l4_list in l4_by_parent.items():
            all_ancestor_ids.add(parent_id)
            for h in l4_list:
                if h.get("root_chunk_id"):
                    all_ancestor_ids.add(h["root_chunk_id"])

        # Fetch ancestors
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

            # Fetch missing L2 ancestors for L3 parents
            missing_l2_ids: set[str] = set()
            for parent_id in l3_ids_from_l4:
                parent_node = ancestor_map.get(parent_id)
                if parent_node and parent_node.get("parent_chunk_id"):
                    missing_l2_ids.add(parent_node["parent_chunk_id"])
            for hit in l3_hits.values():
                if hit["chunk_id"] in l3_ids_from_l4:
                    continue
                if hit.get("parent_chunk_id"):
                    missing_l2_ids.add(hit["parent_chunk_id"])
            missing_l2_ids -= ancestor_map.keys()
            if missing_l2_ids:
                batch = list(missing_l2_ids)
                for i in range(0, len(batch), 100):
                    rows = self.milvus_manager.query_by_chunk_ids(
                        batch[i : i + 100], output_fields=["text", "chunk_id", "chunk_level", "title_path", "parent_chunk_id"]
                    )
                    for row in rows:
                        ancestor_map[row["chunk_id"]] = row

        # --- Phase 3: build context entries ---
        def _build_context(hit: dict) -> dict:
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
                lv = node["chunk_level"]
                if lv == 1:
                    context["chunk1_text"] = node["text"]
                elif lv == 2:
                    context["chunk2_text"] = node["text"]
                elif lv == 3:
                    context["chunk3_text"] = node["text"]
            return context

        # (a) Standalone L3 hits (not covered by L4)
        for cid, hit in l3_hits.items():
            if cid in l3_ids_from_l4:
                continue
            final_contexts.append(_build_context(hit))

        # (b) L4 hits → merge per L3 parent, use L3 text as search_hit
        for parent_id, l4_list in l4_by_parent.items():
            parent_node = ancestor_map.get(parent_id)
            if not parent_node:
                # Can't resolve parent, fall back to first L4 hit
                final_contexts.append(_build_context(l4_list[0]))
                continue

            # Build context from the L3 parent, using first L4 as the base hit
            base_hit = dict(l4_list[0])
            base_hit["text"] = parent_node["text"]
            base_hit["chunk_id"] = parent_id
            base_hit["chunk_level"] = 3
            base_hit["title_path"] = parent_node.get("title_path", base_hit.get("title_path", ""))
            ctx = _build_context(base_hit)
            # Override: search_hit and chunk3_text should be L3 parent text
            ctx["search_hit"] = parent_node["text"]
            ctx["chunk3_text"] = parent_node["text"]
            ctx["chunk4_text"] = ""  # Do not include L4 text
            ctx["chunk_id"] = parent_id
            ctx["level"] = 3
            # Merge image_paths from all L4 hits
            all_images = []
            for h in l4_list:
                ip = h.get("image_paths", "")
                if ip:
                    all_images.append(ip)
            if all_images:
                ctx["image_paths"] = ";".join(dict.fromkeys(all_images))
            final_contexts.append(ctx)

        # (c) Other hits (L2 etc.)
        for hit in other_hits:
            final_contexts.append(_build_context(hit))

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
