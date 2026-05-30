import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import re

from agent.graph import build_graph
from agent.persistence import get_store
from agent.rag.data_embedding import RAGPipelineService
from agent.rag.knowledge_pack import get_pack_manager
from agent.security import check_message
from agent.utils.logger_handler import get_logger

load_dotenv()

COMPRESS_THRESHOLD = 20  # messages
COMPRESS_KEEP = 5        # recent messages to keep

SUMMARY_PROMPT = (
    "请将以下对话历史压缩为一段简洁的摘要，保留关键信息、用户偏好和未完成的任务。"
    "用中文输出，不超过300字。\n\n对话历史：\n{history}")


async def compress_history(messages: list[dict]) -> str:
    """Summarize old messages using DeepSeek V4 Flash."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        timeout=60,
    )
    formatted = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in messages)
    prompt = SUMMARY_PROMPT.format(history=formatted)
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""
from agent.utils.path_handler import get_absolute_path

# Regex to extract chunk_id from tool outputs
# Captures prefix and level_seq separately to preserve spaces in prefix
_CHUNK_ID_RE = re.compile(r"chunk_id[=:]\s*(.*?)(_L\d+_\d+)")

load_dotenv()
logger = get_logger()

# --- Paths ---
DATA_DIR = Path(get_absolute_path("data"))
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_FILE = Path(get_absolute_path("data/knowledge.md"))
STATIC_DIR = Path(get_absolute_path("static"))

# --- App ---
app = FastAPI(title="Study Agent")

# --- Graph (lazy init) ---
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ============================================================
# Knowledge Base
# ============================================================
def _read_knowledge_entries() -> list[dict]:
    """Parse knowledge.md into structured entries."""
    if not KNOWLEDGE_FILE.exists():
        return []
    entries = []
    for line in KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "序号" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 4:
                try:
                    entries.append({
                        "index": int(parts[0]),
                        "filename": parts[1],
                        "chunks": int(parts[2]),
                        "uploaded_at": parts[3],
                    })
                except (ValueError, IndexError):
                    pass
    return entries


def _add_knowledge_entry(filename: str, chunks: int):
    # Strip .md/.markdown suffix for cleaner display
    name = Path(filename).stem if Path(filename).suffix.lower() in (".md", ".markdown") else filename
    if not KNOWLEDGE_FILE.exists():
        KNOWLEDGE_FILE.write_text(
            "# 知识库课本列表\n\n"
            "| 序号 | 文件名 | 片段数 | 上传时间 |\n"
            "|------|--------|--------|----------|\n",
            encoding="utf-8",
        )
    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    existing = [l for l in content.splitlines() if l.startswith("|") and "序号" not in l and "---" not in l]
    idx = len(existing) + 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = f"| {idx} | {name} | {chunks} | {now} |\n"
    if not content.endswith("\n"):
        content += "\n"
    content += new_row
    KNOWLEDGE_FILE.write_text(content, encoding="utf-8")


def _delete_knowledge_entry(index: int) -> bool:
    if not KNOWLEDGE_FILE.exists():
        return False
    lines = KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines()
    new_lines = []
    deleted = False
    for line in lines:
        if line.startswith("|") and "序号" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            try:
                if int(parts[0]) == index:
                    deleted = True
                    continue
            except (ValueError, IndexError):
                pass
        new_lines.append(line)
    if deleted:
        KNOWLEDGE_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return deleted


# ============================================================
# API Routes
# ============================================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    dest = DATA_DIR / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def progress_stream():
        def on_progress(msg: str, current: int, total: int):
            pct = round(current / total * 100) if total > 0 else 0
            progress_queue.put_nowait({"stage": msg, "current": current, "total": total, "percent": pct})

        progress_queue: asyncio.Queue = asyncio.Queue()

        async def run_ingest():
            loop = asyncio.get_event_loop()
            try:
                service = RAGPipelineService(model_name="dashscope")
                try:
                    prepared = await loop.run_in_executor(
                        None, lambda: service.ingest_file(str(dest), on_progress=on_progress)
                    )
                    chunks_count = len(prepared.chunks)
                    _add_knowledge_entry(filename, chunks_count)
                    await progress_queue.put({"stage": "完成", "current": 1, "total": 1, "percent": 100,
                                              "done": True, "filename": filename, "chunks": chunks_count})
                finally:
                    service.close()
            except Exception as e:
                logger.error(f"Ingest error: {e}")
                await progress_queue.put({"stage": f"错误: {e}", "current": 0, "total": 1, "percent": 0,
                                          "done": True, "error": str(e)})

        task = asyncio.create_task(run_ingest())

        # Save file stage
        yield _sse("progress", {"stage": "保存文件", "current": 1, "total": 1, "percent": 5})

        while True:
            try:
                data = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield _sse("progress", data)
                if data.get("done"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                if task.done():
                    break

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/knowledge")
async def get_knowledge():
    return {"entries": _read_knowledge_entries()}


@app.delete("/api/knowledge/{index}")
async def delete_knowledge(index: int):
    deleted = _delete_knowledge_entry(index)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return {"status": "ok"}


@app.get("/api/images/{path:path}")
async def serve_image(path: str):
    from fastapi.responses import FileResponse, Response
    # Try multiple path resolutions
    candidates = [
        DATA_DIR.parent / path,                    # project_root/images/xxx.jpg (legacy)
        DATA_DIR / path,                           # data/images/xxx.jpg
    ]
    # Also try stripping leading "images/" and searching in data subdirs
    filename = Path(path).name
    for book_dir in DATA_DIR.iterdir():
        if book_dir.is_dir():
            candidates.append(book_dir / "images" / filename)

    img_path = None
    for c in candidates:
        if c.exists() and c.is_file():
            img_path = c
            break

    # Fallback: recursive search
    if not img_path:
        matches = list(DATA_DIR.rglob(filename))
        img_path = matches[0] if matches else None

    # Fallback: prefix match for truncated filenames (no extension)
    if not img_path and "." not in filename:
        for ext in ("jpg", "jpeg", "png", "gif", "bmp", "webp", "svg"):
            matches = list(DATA_DIR.rglob(f"{filename}*.{ext}"))
            if matches:
                img_path = matches[0]
                break

    # Path traversal guard: ensure resolved path is within DATA_DIR
    if img_path and not img_path.resolve().is_relative_to(DATA_DIR.resolve()):
        logger.warning(f"[Security] Image path traversal blocked: {path}")
        img_path = None

    if not img_path or not img_path.exists():
        # Return 1x1 transparent PNG instead of 404
        transparent_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return Response(content=transparent_png, media_type="image/png")

    suffix = img_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp", ".svg": "image/svg+xml",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(str(img_path), media_type=media_type)


# ============================================================
# Conversation CRUD
# ============================================================
@app.get("/api/conversations")
async def list_conversations():
    store = get_store()
    return {"conversations": store.list_conversations()}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    store = get_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    store = get_store()
    deleted = store.delete_conversation(conv_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return {"status": "ok"}


@app.delete("/api/conversations/{conv_id}/last")
async def delete_last_messages(conv_id: str):
    store = get_store()
    deleted = store.delete_last_messages(conv_id, count=2)
    return {"status": "ok", "deleted": deleted}


TOOL_NAME_MAP = {
    "knowledge_base_search": "检索知识库",
    "fetch_neighbor_context": "获取上下文片段",
    "view_image": "查看图片",
    "web_search": "搜索互联网",
}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    history = body.get("history", [])
    conv_id = body.get("conv_id", "")
    if not message:
        return JSONResponse(status_code=400, content={"error": "empty message"})

    store = get_store()
    # Auto-create conversation if needed
    if conv_id:
        existing = store.get_conversation(conv_id)
        if not existing:
            store.create_conversation(conv_id, message[:30])
    else:
        conv_id = str(int(datetime.now().timestamp() * 1000))
        store.create_conversation(conv_id, message[:30])

    # Save user message and assistant placeholder
    store.add_message(conv_id, "user", message)
    assistant_msg_id = store.add_message(conv_id, "assistant", "")

    graph = _get_graph()
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": message})

    # Context compression: summarize old messages when too long
    if len(messages) > COMPRESS_THRESHOLD:
        try:
            old = messages[:COMPRESS_THRESHOLD - COMPRESS_KEEP]
            recent = messages[COMPRESS_THRESHOLD - COMPRESS_KEEP:]
            summary = await compress_history(old)
            if summary:
                messages = [{"role": "system", "content": f"[对话摘要] {summary}"}] + recent
                store.update_summary(conv_id, summary)
                logger.info(f"[Compress] Summarized {len(old)} messages -> {len(summary)} chars")
        except Exception as e:
            logger.warning(f"[Compress] Failed (non-fatal): {e}")

    input_msg = {"messages": messages}
    config = {"recursion_limit": 50}

    async def event_stream():
        got_tokens = False
        final_content = ""
        thinking_steps: list[dict] = []
        # Track content length at each tool start — reasoning text accumulates
        # between tools, and the answer is everything after the last tool's position
        last_tool_content_len = 0
        # Pack learning: track chunk IDs that entered LLM context
        final_context_ids: set[str] = set()
        pack_mgr = get_pack_manager()

        try:
            # Security check before invoking the graph
            sec_result = await check_message(message, history)
            if not sec_result.passed:
                yield _sse("token", {"text": sec_result.reason})
                try:
                    store.update_message(assistant_msg_id, sec_result.reason)
                except Exception as e:
                    logger.error(f"Failed to save rejection message: {e}")
                yield _sse("done", {"conv_id": conv_id})
                return

            async for event in graph.astream_events(input_msg, config=config, version="v2"):
                kind = event.get("event", "")

                if kind == "on_tool_start":
                    # Content before this tool call is reasoning
                    last_tool_content_len = len(final_content)
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except Exception:
                            pass
                    display_name = TOOL_NAME_MAP.get(tool_name, tool_name)
                    detail = ""
                    if isinstance(tool_input, dict):
                        for v in tool_input.values():
                            if isinstance(v, str) and len(v) > 2:
                                detail = v[:120]
                                break
                    elif isinstance(tool_input, str):
                        detail = tool_input[:120]
                    thinking_steps.append({"tool": display_name, "detail": detail, "result": ""})
                    yield _sse("thinking", {"tool": display_name, "detail": detail})

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    if hasattr(output, "content"):
                        output = output.content or ""
                    if not isinstance(output, str):
                        output = str(output)

                    # Extract chunk IDs from FULL output before truncation
                    if output:
                        # Reconstruct full chunk IDs from match groups (preserves spaces in prefix)
                        matches = _CHUNK_ID_RE.findall(output)
                        found_ids = [f"{prefix}{level_seq}" for prefix, level_seq in matches]
                        logger.info(f"[Pack-Debug] tool={event.get('name','')}, output_len={len(output)}, found_ids={found_ids[:5]}")
                        if found_ids:
                            final_context_ids.update(found_ids)
                            pack_mgr.record_expansion(found_ids)

                    display_output = output[:200] if isinstance(output, str) else str(output)[:200]
                    if thinking_steps:
                        thinking_steps[-1]["result"] = display_output
                    yield _sse("tool_result", {"output": display_output})

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    text = ""
                    if isinstance(chunk, dict):
                        text = chunk.get("content", "")
                    elif hasattr(chunk, "content"):
                        text = chunk.content or ""
                    if text:
                        got_tokens = True
                        final_content += text
                        yield _sse("token", {"text": text})

                elif kind == "on_chat_model_end":
                    # Only use as fallback if no tokens were streamed
                    if not got_tokens:
                        output = event.get("data", {}).get("output", {})
                        if hasattr(output, "content"):
                            final_content = output.content or ""
                        elif isinstance(output, dict):
                            final_content = output.get("content", "")

                elif kind == "on_chain_end":
                    # Capture from chain output if still no tokens
                    if not got_tokens and not final_content:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            msgs = output.get("messages", [])
                            if msgs:
                                last = msgs[-1]
                                if hasattr(last, "content"):
                                    final_content = last.content or ""
                                elif isinstance(last, dict):
                                    final_content = last.get("content", "")

        except Exception as e:
            # Check if it's a recursion limit error
            err_str = str(e)
            if "Recursion limit" in err_str or "GRAPH_RECURSION_LIMIT" in err_str:
                logger.warning(f"Graph recursion limit reached")
                yield _sse("recursion_limit", {"message": "思考步骤较多，已暂停。点击「继续」让 AI 继续回答。"})
            else:
                logger.error(f"Stream error: {e}", exc_info=True)
                yield _sse("error", {"message": str(e)})

        # Fallback: if no tokens were streamed but we have final content
        if not got_tokens and final_content:
            logger.warning("No streaming tokens received, using fallback final content")
            chunk_size = 10
            for i in range(0, len(final_content), chunk_size):
                yield _sse("token", {"text": final_content[i:i + chunk_size]})

        # Send thinking_done with the split position (content before last tool = reasoning)
        if thinking_steps:
            yield _sse("thinking_done", {"steps": len(thinking_steps), "split": last_tool_content_len})

        # Update assistant message with final content
        answer = final_content or ""
        try:
            store.update_message(
                assistant_msg_id, answer,
                thinking=thinking_steps if thinking_steps else None,
            )
        except Exception as e:
            logger.error(f"Failed to save message: {e}")

        # Pack learning: finalize packs from this conversation
        try:
            pack_mgr.finalize_packs(final_context_ids)
        except Exception as e:
            logger.error(f"Failed to finalize packs: {e}")

        yield _sse("done", {"conv_id": conv_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- Serve frontend ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Evaluation endpoints ---
@app.get("/eval", response_class=HTMLResponse)
async def eval_page():
    return (STATIC_DIR / "eval.html").read_text(encoding="utf-8")


@app.post("/api/eval/run")
async def eval_run(request: Request):
    from agent.evaluation import EvalRunner

    body = await request.json()
    test_cases = body.get("test_cases", [])
    if not test_cases:
        return JSONResponse(status_code=400, content={"error": "test_cases is required"})

    runner = EvalRunner()

    async def event_stream():
        try:
            q: asyncio.Queue[str | None] = asyncio.Queue()

            async def on_progress_cb(idx, total, question):
                await q.put(_sse("progress", {"current": idx + 1, "total": total, "question": question}))

            async def run_eval():
                try:
                    result = await runner.run_evaluation(test_cases, on_progress=on_progress_cb)
                    # Persist results
                    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    eval_path = DATA_DIR / "eval_results"
                    eval_path.mkdir(exist_ok=True)
                    result_file = eval_path / f"{eval_id}.json"
                    result_file.write_text(json.dumps({"test_cases": test_cases, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
                    logger.info(f"[Eval] Results saved to {result_file}")
                    await q.put(_sse("result", result))
                    await q.put(None)  # sentinel
                except Exception as e:
                    logger.error(f"Eval task error: {e}", exc_info=True)
                    await q.put(_sse("error", {"message": str(e)}))
                    await q.put(None)

            task = asyncio.create_task(run_eval())

            while True:
                msg = await q.get()
                if msg is None:
                    break
                yield msg

            # Check if task had an unhandled exception
            if task.exception():
                yield _sse("error", {"message": str(task.exception())})

        except Exception as e:
            logger.error(f"Eval error: {e}", exc_info=True)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/eval/results")
async def eval_results():
    """List saved evaluation results."""
    eval_path = DATA_DIR / "eval_results"
    if not eval_path.exists():
        return JSONResponse(content=[])
    files = sorted(eval_path.glob("*.json"), reverse=True)
    results = []
    for f in files[:20]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "id": f.stem,
                "scores": data.get("scores", {}),
                "total": len(data.get("test_cases", [])),
            })
        except Exception:
            pass
    return JSONResponse(content=results)


@app.get("/api/eval/results/{eval_id}")
async def eval_result_detail(eval_id: str):
    """Get a specific evaluation result."""
    eval_path = DATA_DIR / "eval_results" / f"{eval_id}.json"
    if not eval_path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8080, reload=True)
