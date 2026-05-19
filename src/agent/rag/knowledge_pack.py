import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent.rag.milvus_manage import MilvusManage
from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_absolute_path

logger = get_logger()

DB_PATH = str(get_absolute_path("data/chat.db"))

# Limits
MAX_CHUNKS_PER_PACK = 8
MAX_PACKS_PER_L2 = 5
DEDUP_JACCARD_THRESHOLD = 0.8


def _parse_chunk_id(chunk_id: str) -> tuple[str, int, int] | tuple[None, None, None]:
    try:
        parts = chunk_id.rsplit("_L", 1)
        if len(parts) != 2:
            return None, None, None
        level_seq = parts[1].split("_", 1)
        if len(level_seq) != 2:
            return None, None, None
        return parts[0], int(level_seq[0]), int(level_seq[1])
    except (ValueError, IndexError):
        return None, None, None


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class KnowledgePackManager:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._milvus: MilvusManage | None = None
        # Per-conversation buffer: l2_chunk_id -> set of expanded chunk_ids
        self._buffer: dict[str, set[str]] = defaultdict(set)
        # Track pack IDs touched (created/merged/hit) in current conversation
        self._touched_pack_ids: set[int] = set()
        self._init_tables()

    def _get_milvus(self) -> MilvusManage:
        if self._milvus is None:
            self._milvus = MilvusManage()
        return self._milvus

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                l2_chunk_id TEXT NOT NULL,
                chunk_ids TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_packs_l2 ON knowledge_packs(l2_chunk_id);
        """)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    def record_expansion(self, chunk_ids: list[str]):
        """
        Record chunks from a single tool call.
        Groups by L2 parent and accumulates in buffer.
        """
        if not chunk_ids:
            return

        logger.info(f"[Pack] record_expansion: {len(chunk_ids)} chunk(s): {chunk_ids[:5]}{'...' if len(chunk_ids)>5 else ''}")

        # Resolve L2 parent for each chunk
        milvus = self._get_milvus()
        rows = milvus.query_by_chunk_ids(
            chunk_ids,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level"],
        )
        row_map = {r["chunk_id"]: r for r in rows}
        logger.info(f"[Pack] Milvus returned {len(rows)}/{len(chunk_ids)} rows")

        for cid in chunk_ids:
            row = row_map.get(cid)
            if not row:
                logger.debug(f"[Pack] No Milvus row for {cid}")
                continue

            level = row.get("chunk_level", 0)
            parent_id = row.get("parent_chunk_id", "")

            if level == 2:
                l2_id = cid
            elif level == 3:
                l2_id = parent_id if parent_id else cid
            elif level == 4:
                if parent_id:
                    l3_row = milvus.query_by_chunk_ids(
                        [parent_id],
                        output_fields=["parent_chunk_id", "chunk_level"],
                    )
                    if l3_row and l3_row[0].get("chunk_level") == 3:
                        l2_id = l3_row[0].get("parent_chunk_id", parent_id)
                    else:
                        l2_id = parent_id
                else:
                    continue
            else:
                continue

            if l2_id:
                self._buffer[l2_id].add(cid)

        logger.info(f"[Pack] Buffer state: { {k: len(v) for k, v in self._buffer.items()} }")

    def finalize_packs(self, final_context_ids: set[str]):
        """
        Called at end of conversation. Writes packs to DB.
        """
        logger.info(f"[Pack] finalize_packs: buffer has {len(self._buffer)} L2 group(s), final_context_ids has {len(final_context_ids)} chunk(s)")
        if not self._buffer:
            return

        now = datetime.now().isoformat(timespec="seconds")

        for l2_id, expanded_ids in self._buffer.items():
            logger.info(f"[Pack] L2={l2_id}: {len(expanded_ids)} chunk(s) in buffer: {sorted(expanded_ids)}")
            # Only record if multiple chunks in same L2
            if len(expanded_ids) < 2:
                logger.info(f"[Pack] Skipping L2={l2_id}: only {len(expanded_ids)} chunk(s)")
                continue

            # Filter to only chunks that entered final LLM context
            used_ids = [cid for cid in expanded_ids if cid in final_context_ids]
            logger.info(f"[Pack] L2={l2_id}: {len(used_ids)}/{len(expanded_ids)} in final_context_ids")
            if len(used_ids) < 2:
                logger.info(f"[Pack] Skipping L2={l2_id}: only {len(used_ids)} used chunk(s)")
                continue

            # Enforce per-pack chunk limit
            used_ids = used_ids[:MAX_CHUNKS_PER_PACK]

            self._merge_or_create_pack(l2_id, used_ids, now)

        self._decay_unused_packs()
        self.conn.commit()
        self._buffer.clear()
        self._touched_pack_ids.clear()

    def _merge_or_create_pack(self, l2_id: str, chunk_ids: list[str], now: str):
        """Merge with existing pack if similar, otherwise create new."""
        new_set = set(chunk_ids)

        # Get existing packs for this L2
        existing = self.conn.execute(
            "SELECT id, chunk_ids, hit_count FROM knowledge_packs WHERE l2_chunk_id = ?",
            (l2_id,),
        ).fetchall()

        # Check pack count limit
        if len(existing) >= MAX_PACKS_PER_L2:
            # Find the pack with lowest hit_count to potentially replace
            existing.sort(key=lambda r: r["hit_count"])
            lowest = existing[0]
            lowest_set = set(json.loads(lowest["chunk_ids"]))
            if _jaccard(new_set, lowest_set) >= DEDUP_JACCARD_THRESHOLD:
                # Merge with the lowest-hit pack
                merged = sorted(lowest_set | new_set)[:MAX_CHUNKS_PER_PACK]
                self.conn.execute(
                    "UPDATE knowledge_packs SET chunk_ids=?, chunk_count=?, hit_count=hit_count+1, updated_at=? WHERE id=?",
                    (json.dumps(merged), len(merged), now, lowest["id"]),
                )
                self._touched_pack_ids.add(lowest["id"])
                logger.info(f"[Pack] Merged into L2={l2_id} pack #{lowest['id']}: {len(lowest_set)}->{len(merged)} chunks")
                return
            else:
                # Replace the lowest-hit pack
                self.conn.execute(
                    "UPDATE knowledge_packs SET chunk_ids=?, chunk_count=?, hit_count=2, updated_at=? WHERE id=?",
                    (json.dumps(chunk_ids), len(chunk_ids), now, lowest["id"]),
                )
                self._touched_pack_ids.add(lowest["id"])
                logger.info(f"[Pack] Replaced low-hit pack #{lowest['id']} in L2={l2_id}")
                return

        # Check if we can merge with any existing pack
        for row in existing:
            existing_set = set(json.loads(row["chunk_ids"]))
            if _jaccard(new_set, existing_set) >= DEDUP_JACCARD_THRESHOLD:
                merged = sorted(existing_set | new_set)[:MAX_CHUNKS_PER_PACK]
                self.conn.execute(
                    "UPDATE knowledge_packs SET chunk_ids=?, chunk_count=?, hit_count=hit_count+1, updated_at=? WHERE id=?",
                    (json.dumps(merged), len(merged), now, row["id"]),
                )
                self._touched_pack_ids.add(row["id"])
                logger.info(f"[Pack] Merged with pack #{row['id']} in L2={l2_id}: {len(existing_set)}->{len(merged)} chunks")
                return

        # Create new pack with hit_count=2 (survives one round of decay)
        cur = self.conn.execute(
            "INSERT INTO knowledge_packs (l2_chunk_id, chunk_ids, chunk_count, hit_count, created_at, updated_at) VALUES (?, ?, ?, 2, ?, ?)",
            (l2_id, json.dumps(chunk_ids), len(chunk_ids), now, now),
        )
        self._touched_pack_ids.add(cur.lastrowid)
        logger.info(f"[Pack] Created new pack in L2={l2_id}: {len(chunk_ids)} chunks")

    def get_packs_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """
        Find packs that contain any of the given chunk_ids.
        Returns {matched_chunk_id: [other chunk_ids in the same pack]}.
        """
        if not chunk_ids:
            return {}

        result: dict[str, list[str]] = {}
        # Query all packs that might contain these chunks
        # Since chunk_ids is a JSON array, we need to check each one
        # For efficiency, first find which L2 parents these chunks belong to
        l2_ids = self._resolve_l2_parents(chunk_ids)

        if not l2_ids:
            return {}

        # Query packs by L2
        placeholders = ",".join("?" * len(l2_ids))
        rows = self.conn.execute(
            f"SELECT id, l2_chunk_id, chunk_ids, hit_count FROM knowledge_packs WHERE l2_chunk_id IN ({placeholders})",
            list(l2_ids),
        ).fetchall()

        for row in rows:
            pack_chunks = set(json.loads(row["chunk_ids"]))
            matched = pack_chunks & set(chunk_ids)
            if matched:
                others = sorted(pack_chunks - set(chunk_ids))
                if others:
                    for mid in matched:
                        result[mid] = others
                    # Record hit and mark as touched
                    self.conn.execute(
                        "UPDATE knowledge_packs SET hit_count = hit_count + 1 WHERE id = ?",
                        (row["id"],),
                    )
                    self._touched_pack_ids.add(row["id"])

        if result:
            self.conn.commit()

        return result

    def _resolve_l2_parents(self, chunk_ids: list[str]) -> set[str]:
        """Resolve L2 parent IDs for a list of chunk IDs."""
        l2_ids: set[str] = set()
        need_query: list[str] = []

        for cid in chunk_ids:
            prefix, level, seq = _parse_chunk_id(cid)
            if level is None:
                continue
            if level == 2:
                l2_ids.add(cid)
            elif level in (3, 4):
                need_query.append(cid)

        if need_query:
            milvus = self._get_milvus()
            rows = milvus.query_by_chunk_ids(
                need_query,
                output_fields=["chunk_id", "parent_chunk_id", "chunk_level"],
            )
            row_map = {r["chunk_id"]: r for r in rows}

            # Collect L3 parent IDs for L4 chunks
            l3_parents: list[str] = []
            for cid in need_query:
                row = row_map.get(cid)
                if not row:
                    continue
                if row["chunk_level"] == 3:
                    parent = row.get("parent_chunk_id", "")
                    if parent:
                        l2_ids.add(parent)
                elif row["chunk_level"] == 4:
                    parent = row.get("parent_chunk_id", "")
                    if parent:
                        l3_parents.append(parent)

            # Resolve L3 -> L2
            if l3_parents:
                l3_rows = milvus.query_by_chunk_ids(
                    l3_parents,
                    output_fields=["chunk_id", "parent_chunk_id", "chunk_level"],
                )
                for r in l3_rows:
                    if r.get("chunk_level") == 3 and r.get("parent_chunk_id"):
                        l2_ids.add(r["parent_chunk_id"])

        return l2_ids

    def _decay_unused_packs(self):
        """Decrease hit_count for packs not touched in this session. Those reaching 0 are deleted."""
        if self._touched_pack_ids:
            placeholders = ",".join("?" * len(self._touched_pack_ids))
            self.conn.execute(
                f"UPDATE knowledge_packs SET hit_count = hit_count - 1 WHERE hit_count > 0 AND id NOT IN ({placeholders})",
                list(self._touched_pack_ids),
            )
        else:
            self.conn.execute(
                "UPDATE knowledge_packs SET hit_count = hit_count - 1 WHERE hit_count > 0"
            )
        deleted = self.conn.execute(
            "DELETE FROM knowledge_packs WHERE hit_count <= 0"
        ).rowcount
        if deleted:
            logger.info(f"[Pack] Decayed and removed {deleted} unused pack(s)")

    def invalidate_by_doc(self, doc_id: str):
        """Delete all packs whose L2 chunks belong to the given document."""
        # Find all L2 chunk_ids for this doc
        milvus = self._get_milvus()
        prefix = doc_id[:8]
        # L2 IDs follow pattern: {prefix}_L2_*
        # We need to find them via query
        rows = milvus.query_by_chunk_ids(
            # Can't use wildcard, so we query by doc_id filter
            [],  # empty, will use filter instead
        )
        # Actually, we need a different approach - query by doc_id
        # For now, use a simpler approach: delete packs where l2_chunk_id starts with prefix
        self.conn.execute(
            "DELETE FROM knowledge_packs WHERE l2_chunk_id LIKE ?",
            (f"{prefix}%",),
        )
        self.conn.commit()
        logger.info(f"[Pack] Invalidated packs for doc {doc_id} (prefix={prefix})")


# Singleton
_pack_manager: KnowledgePackManager | None = None


def get_pack_manager() -> KnowledgePackManager:
    global _pack_manager
    if _pack_manager is None:
        _pack_manager = KnowledgePackManager()
    return _pack_manager
