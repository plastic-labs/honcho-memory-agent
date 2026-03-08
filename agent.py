"""
Always-On Memory Agent — Honcho Version

Honcho replaces the entire manual memory stack from the Google ADK reference:

  Reference concept             Honcho primitive
  ────────────────────────────  ─────────────────────────────────────────────
  store_memory() + SQLite       session.add_messages() -> Deriver (bg)
  entity/topic extraction       Deriver agent (LLM-based, automatic)
  consolidate_agent + loop      Dreamer (automatic + POST /schedule_dream)
  read_all_memories()           peer.representation()
  query_agent + orchestrator    peer.chat()  (Dialectic API)
  processed_files table         message metadata (source_file) + local set
  importance scores             observation levels (explicit / deductive)

This script is a thin wrapper that adds:
  - File watching (drop files into ./inbox to ingest)
  - Peer attribution (filename convention, .meta.json sidecar, API-level)
  - Media preprocessing (LLM description for images before Honcho ingestion)
  - HTTP API proxying to Honcho SDK calls

Usage:
    python agent.py
    python agent.py --watch ./docs --port 9000 --peer alice

Endpoints:
    GET  /query?q=         -> peer.chat()          (Dialectic)
    POST /ingest           -> session.add_messages() or upload_file()
    GET  /memories?q=      -> peer.representation()
    POST /consolidate      -> honcho.schedule_dream()
    GET  /status           -> peer card + session summaries
    POST /clear            -> session.delete()

Environment:
    HONCHO_API_KEY          API key (required for hosted; omit for local)
    HONCHO_ENV              "demo" | "local" | custom base URL (default: demo)
    HONCHO_WORKSPACE_ID     workspace slug (default: always-on-agent)
    HONCHO_DEFAULT_PEER     default peer ID when none resolved (default: user)
    ANTHROPIC_API_KEY       for image description preprocessing
    MEDIA_MODEL             model for image description (default: claude-haiku-4-5-20251001)
"""

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from aiohttp import web
from dotenv import load_dotenv

from honcho import Honcho

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

HONCHO_ENV = os.getenv("HONCHO_ENV", "demo")
HONCHO_API_KEY = os.getenv("HONCHO_API_KEY", "")
WORKSPACE_ID = os.getenv("HONCHO_WORKSPACE_ID", "always-on-agent")
DEFAULT_PEER_ID = os.getenv("HONCHO_DEFAULT_PEER", "user")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MEDIA_MODEL = os.getenv("MEDIA_MODEL", "claude-haiku-4-5-20251001")

# Honcho natively extracts text from these via the file upload endpoint.
HONCHO_NATIVE_TYPES = {
    ".txt", ".md", ".json", ".csv", ".log",
    ".xml", ".yaml", ".yml", ".pdf",
}

# These need LLM-based description before Honcho ingestion.
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Audio/video require a multimodal model with AV support (e.g. Gemini Flash).
# See open question in spec — not implemented here by default.
AV_TYPES = {
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac",
    ".mp4", ".webm", ".mov", ".avi", ".mkv",
}

MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".pdf": "application/pdf",
    ".json": "application/json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="[%H:%M]")
log = logging.getLogger("honcho-memory-agent")

# Module-level Anthropic client — instantiated once, reused across all image calls.
_anthropic_client: anthropic.AsyncAnthropic | None = None


def get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


# ── Attribution ────────────────────────────────────────────────────────────────


def daily_session_id() -> str:
    return f"inbox-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


def resolve_attribution(f: Path, default_peer: str) -> tuple[str, str, str | None]:
    """
    Resolve peer, session, and optional context from a file path.

    Priority:
      1. Sidecar .meta.json  e.g. meeting-notes.meta.json
      2. Filename convention  e.g. alice--meeting-notes.md  -> peer=alice
      3. Default fallback     configured default_peer + daily session
    """
    sidecar = f.parent / (f.stem + ".meta.json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            peer = meta.get("peer", default_peer)
            session = meta.get("session") or daily_session_id()
            context = meta.get("context")
            return peer, session, context
        except Exception as exc:
            log.warning("could not parse sidecar %s: %s", sidecar.name, exc)

    if "--" in f.stem:
        peer = f.stem.split("--", 1)[0]
        return peer, daily_session_id(), None

    return default_peer, daily_session_id(), None


# ── Media Preprocessing ────────────────────────────────────────────────────────


async def preprocess_media(
    file_path: Path,
    source_peer: str,
    context: str | None = None,
) -> str:
    """
    Convert a media file into a rich textual description for Honcho ingestion.

    Images are described via Claude Haiku (multimodal).
    Audio/video preprocessing requires an AV-capable model and is not
    implemented here; a metadata stub is returned instead.

    Args:
        file_path:   Path to the media file.
        source_peer: Peer this content is attributed to (used in prompt framing).
        context:     Optional hint, e.g. "screenshot from a design review".

    Returns:
        Textual description with attribution metadata prefix.
    """
    suffix = file_path.suffix.lower()
    mime_type = MIME_MAP.get(suffix) or (mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")

    if suffix in IMAGE_TYPES:
        return await _describe_image(file_path, mime_type, source_peer, context)

    if suffix in AV_TYPES:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        log.warning(
            "audio/video preprocessing not implemented; ingesting metadata stub for %s",
            file_path.name,
        )
        context_line = f"Context: {context}\n" if context else ""
        return (
            f"[Media: {file_path.name}  type: {mime_type}  size: {size_mb:.1f}MB  source: {source_peer}]\n"
            f"{context_line}"
            "Audio/video description requires an AV-capable model (e.g. Gemini Flash). "
            "Configure MEDIA_MODEL and extend _describe_av() to enable full preprocessing."
        )

    raise ValueError(f"preprocess_media called on unsupported type: {suffix}")


async def _describe_image(
    file_path: Path,
    mime_type: str,
    source_peer: str,
    context: str | None,
) -> str:
    file_bytes = file_path.read_bytes()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > 20:
        log.warning("skipping %s (%.1f MB exceeds 20 MB limit)", file_path.name, size_mb)
        return f"[Image skipped: {file_path.name} is {size_mb:.1f} MB, exceeds limit]"

    context_line = f"Context: {context}\n" if context else ""
    prompt = (
        f"Describe this image thoroughly for memory storage.\n"
        f"Source peer: {source_peer}\n"
        f"Filename: {file_path.name}\n"
        f"{context_line}"
        "Include: what is depicted, any visible text, people, objects, actions, and "
        "relevant details. Prioritize facts over interpretation."
    )

    client = get_anthropic_client()
    response = await client.messages.create(
        model=MEDIA_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64.standard_b64encode(file_bytes).decode(),
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    description = response.content[0].text
    return (
        f"[Image: {file_path.name}  type: {mime_type}  source: {source_peer}]\n"
        f"{context_line}"
        f"{description}"
    )


# ── Agent ──────────────────────────────────────────────────────────────────────


class HonchoMemoryAgent:
    """
    Thin client over Honcho. All memory operations delegate to Honcho:
      - ingest     -> session.add_messages() or session.upload_file()
      - query      -> peer.chat()            (Dialectic API)
      - memories   -> peer.representation()
      - consolidate -> honcho.schedule_dream()
      - status     -> peer.get_card() + session.summaries()
    """

    def __init__(self, default_peer: str = DEFAULT_PEER_ID) -> None:
        self.honcho = Honcho(
            environment=HONCHO_ENV,
            api_key=HONCHO_API_KEY,
            workspace_id=WORKSPACE_ID,
        )
        self.default_peer_id = default_peer

    def _peer(self, peer_id: str):
        return self.honcho.peer(peer_id)

    def _session(self, session_id: str):
        return self.honcho.session(session_id)

    async def ingest_text(
        self,
        text: str,
        peer_id: str,
        session_id: str,
        source: str = "",
        context: str | None = None,
    ) -> str:
        peer = self._peer(peer_id)
        session = self._session(session_id)

        lines: list[str] = []
        if source:
            lines.append(f"[source: {source}]")
        if context:
            lines.append(f"[context: {context}]")
        lines.append(text)
        content = "\n".join(lines)

        metadata: dict[str, object] = {}
        if source:
            metadata["source_file"] = source

        msg = peer.message(content, metadata=metadata if metadata else None)
        await session.aio.add_messages([msg])
        log.info("ingested text from %r into %s/%s (%d chars)", source or "api", peer_id, session_id, len(text))
        return f"ingested {len(text)} chars into peer={peer_id} session={session_id}"

    async def ingest_file(
        self,
        file_path: Path,
        peer_id: str,
        session_id: str,
        context: str | None = None,
    ) -> str:
        suffix = file_path.suffix.lower()
        peer = self._peer(peer_id)
        session = self._session(session_id)
        mime_type = MIME_MAP.get(suffix) or (mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
        metadata: dict[str, object] = {"source_file": file_path.name}

        if suffix in HONCHO_NATIVE_TYPES:
            # Honcho extracts text natively via the file upload endpoint.
            with open(file_path, "rb") as fh:
                await session.aio.upload_file(
                    (file_path.name, fh.read(), mime_type),
                    peer=peer,
                    metadata=metadata,
                )
            log.info("uploaded %s (native) into %s/%s", file_path.name, peer_id, session_id)
            return f"uploaded {file_path.name} (native extraction) into peer={peer_id}"

        if suffix in IMAGE_TYPES or suffix in AV_TYPES:
            text = await preprocess_media(file_path, source_peer=peer_id, context=context)
            msg = peer.message(text, metadata=metadata)
            await session.aio.add_messages([msg])
            log.info("ingested %s (preprocessed) into %s/%s", file_path.name, peer_id, session_id)
            return f"ingested {file_path.name} (preprocessed) into peer={peer_id}"

        log.warning("unsupported file type %s, skipping", suffix)
        return f"skipped {file_path.name}: unsupported type {suffix}"

    async def query(self, question: str, peer_id: str, session_id: str | None = None) -> str:
        peer = self._peer(peer_id)
        session = self._session(session_id) if session_id else None
        answer = await peer.aio.chat(question, session=session)
        return answer or "no relevant information found"

    async def memories(
        self,
        peer_id: str,
        session_id: str | None = None,
        search_query: str | None = None,
    ) -> str:
        peer = self._peer(peer_id)
        session = self._session(session_id) if session_id else None
        return await peer.aio.representation(session=session, search_query=search_query)

    async def consolidate(self, peer_id: str, session_id: str | None = None) -> str:
        session = self._session(session_id) if session_id else None
        await self.honcho.aio.schedule_dream(observer=peer_id, session=session)
        log.info("dream scheduled for peer=%s session=%s", peer_id, session_id or "global")
        return f"consolidation scheduled for peer={peer_id}"

    async def status(self, peer_id: str, session_id: str) -> dict:
        peer = self._peer(peer_id)
        session = self._session(session_id)
        card = await peer.aio.get_card()
        summaries = await session.aio.summaries()
        return {
            "workspace_id": WORKSPACE_ID,
            "peer_id": peer_id,
            "session_id": session_id,
            "peer_card": card,
            "short_summary": summaries.short_summary.content if summaries.short_summary else None,
            "long_summary": summaries.long_summary.content if summaries.long_summary else None,
        }

    async def clear(self, session_id: str) -> str:
        session = self._session(session_id)
        await session.aio.delete()
        log.info("cleared session %s", session_id)
        return f"deleted session={session_id}"


# ── File Watcher ───────────────────────────────────────────────────────────────

# Tracks files ingested in this process lifetime. Not persisted across restarts —
# files dropped while the agent is down will be re-ingested on next startup.
# For persistence, query Honcho message metadata for existing source_file entries.
_processed: set[str] = set()


async def watch_folder(
    agent: HonchoMemoryAgent,
    folder: Path,
    default_peer: str,
    poll_interval: int = 5,
) -> None:
    """Watch a folder for new files and ingest them into Honcho."""
    folder.mkdir(parents=True, exist_ok=True)
    all_supported = HONCHO_NATIVE_TYPES | IMAGE_TYPES | AV_TYPES
    log.info("watching: %s/", folder)

    while True:
        try:
            for f in sorted(folder.iterdir()):
                if f.name.startswith(".") or f.suffix.lower() == ".meta.json":
                    continue
                if str(f) in _processed:
                    continue
                if f.suffix.lower() not in all_supported:
                    continue

                peer_id, session_id, context = resolve_attribution(f, default_peer)
                try:
                    await agent.ingest_file(f, peer_id=peer_id, session_id=session_id, context=context)
                    _processed.add(str(f))
                except Exception as exc:
                    log.error("error ingesting %s: %s — will retry next poll", f.name, exc)

        except Exception as exc:
            log.error("watch error: %s", exc)

        await asyncio.sleep(poll_interval)


# ── HTTP API ───────────────────────────────────────────────────────────────────


def build_http(agent: HonchoMemoryAgent, default_peer: str) -> web.Application:
    app = web.Application()

    async def handle_query(req: web.Request) -> web.Response:
        q = req.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        peer_id = req.query.get("peer", default_peer)
        session_id = req.query.get("session") or None
        answer = await agent.query(q, peer_id=peer_id, session_id=session_id)
        return web.json_response({"question": q, "answer": answer, "peer": peer_id})

    async def handle_ingest(req: web.Request) -> web.Response:
        """
        POST /ingest
        Body: { "text": "...", "peer": "alice", "session": "...", "source": "...", "context": "..." }
        peer, session, source, context are optional.
        """
        try:
            data = await req.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        peer_id = data.get("peer", default_peer)
        session_id = data.get("session") or daily_session_id()
        source = data.get("source", "api")
        context = data.get("context")
        result = await agent.ingest_text(
            text, peer_id=peer_id, session_id=session_id, source=source, context=context
        )
        return web.json_response({"status": "ingested", "result": result})

    async def handle_memories(req: web.Request) -> web.Response:
        peer_id = req.query.get("peer", default_peer)
        session_id = req.query.get("session") or None
        search_query = req.query.get("q") or None
        rep = await agent.memories(peer_id=peer_id, session_id=session_id, search_query=search_query)
        return web.json_response({"peer": peer_id, "representation": rep})

    async def handle_consolidate(req: web.Request) -> web.Response:
        try:
            data = await req.json()
        except Exception:
            data = {}
        peer_id = data.get("peer", default_peer)
        session_id = data.get("session") or None
        result = await agent.consolidate(peer_id=peer_id, session_id=session_id)
        return web.json_response({"status": "scheduled", "result": result})

    async def handle_status(req: web.Request) -> web.Response:
        peer_id = req.query.get("peer", default_peer)
        session_id = req.query.get("session") or daily_session_id()
        return web.json_response(await agent.status(peer_id=peer_id, session_id=session_id))

    async def handle_clear(req: web.Request) -> web.Response:
        try:
            data = await req.json()
        except Exception:
            data = {}
        session_id = data.get("session") or daily_session_id()
        result = await agent.clear(session_id=session_id)
        return web.json_response({"status": "cleared", "result": result})

    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_get("/memories", handle_memories)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/clear", handle_clear)

    return app


# ── Main ───────────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    agent = HonchoMemoryAgent(default_peer=args.peer)

    log.info("honcho memory agent starting")
    log.info("  workspace : %s", WORKSPACE_ID)
    log.info("  peer      : %s", args.peer)
    log.info("  session   : %s (daily, rotating)", daily_session_id())
    log.info("  watch     : %s", args.watch)
    log.info("  api       : http://localhost:%d", args.port)

    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch), default_peer=args.peer))
    ]

    app = build_http(agent, default_peer=args.peer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info("ready")
    log.info("  drop files in %s/  or  POST to /ingest", args.watch)
    log.info("  attribution: {peer}--{name}.{ext}  or  {name}.meta.json sidecar")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Always-On Memory Agent (Honcho)")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch (default: ./inbox)")
    parser.add_argument("--port", type=int, default=8888, help="HTTP port (default: 8888)")
    parser.add_argument("--peer", default=DEFAULT_PEER_ID, help="Default peer ID (default: user)")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()

    def shutdown(_sig: int) -> None:
        log.info("shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()
        log.info("stopped.")


if __name__ == "__main__":
    main()
