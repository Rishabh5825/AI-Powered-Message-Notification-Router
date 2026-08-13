"""
Stage 3 — Contextual Indexer

Implements Contextual Retrieval: for each historical message in
message_history.csv, generates a short contextual header using the
local LLM, then embeds the combined chunk (header + raw content)
into a vector store.

This is a **one-time offline batch job** — run once before inference.
Results are persisted to disk (vector store + contextual_chunks.json).
"""

import json
import os

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBED_MODEL,
    VECTOR_STORE_DIR,
    CONTEXTUAL_CHUNKS_PATH,
    CACHE_DIR,
    CONTEXT_HEADER_MAX_TOKENS,
)
from data_loader import DataStore


# ──────────────────────────────────────────────────────────────────
# Header Generation Prompt
# ──────────────────────────────────────────────────────────────────

HEADER_PROMPT_TEMPLATE = """You are given a single historical WhatsApp message and its surrounding context.
Generate a concise 2-3 sentence contextual header that explains what a reader
needs to understand this message without seeing the rest of the conversation.

Include: who sent it, the relationship/group type, the topic, and how the
user reacted (opened, dismissed, reported, etc.) if known.

Surrounding context (recent messages in the same thread):
{thread_context}

User's reaction to this message:
{event_summary}

The message:
Sender: {sender_id} | Conversation: {conv_type} | Group/Business: {entity_id}
Timestamp: {timestamp}
Text: {message_text}

Respond with ONLY the contextual header, nothing else."""


class ContextualIndexer:
    """
    Build a vector index of contextual embeddings for message_history.csv.

    Pipeline per historical message:
      1. Gather surrounding thread context (last 3-5 messages from same sender/group)
      2. Look up user reaction from message_events.csv
      3. Call local LLM to generate a contextual header
      4. Combine: contextualized_chunk = header + "\\n" + raw_chunk
      5. Embed with local embedding model
      6. Store in vector DB, tagged with user_id, sender/group/business IDs
    """

    def __init__(self, store: DataStore):
        self.store = store
        self._ollama_client = None
        self._vector_store = None
        self._chunks_cache = self._load_chunks_cache()

    # ──────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────

    def _load_chunks_cache(self) -> dict:
        """Load previously generated contextual chunks."""
        if CONTEXTUAL_CHUNKS_PATH.exists():
            with open(CONTEXTUAL_CHUNKS_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_chunks_cache(self):
        """Persist contextual chunks to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONTEXTUAL_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self._chunks_cache, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────
    # Ollama Client
    # ──────────────────────────────────────────────────────────────

    def _get_ollama_client(self):
        """Lazy-init Ollama client."""
        if self._ollama_client is None:
            try:
                import ollama
                self._ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._ollama_client

    def _generate_header(self, prompt: str) -> str:
        """Call the local LLM to generate a contextual header."""
        client = self._get_ollama_client()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": CONTEXT_HEADER_MAX_TOKENS},
        )
        return response["message"]["content"].strip()

    def _embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a text string using the local model."""
        client = self._get_ollama_client()
        response = client.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]

    # ──────────────────────────────────────────────────────────────
    # Vector Store
    # ──────────────────────────────────────────────────────────────

    def _get_vector_store(self):
        """Lazy-init ChromaDB collection."""
        if self._vector_store is None:
            try:
                import chromadb

                VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
                chroma_client = chromadb.PersistentClient(
                    path=str(VECTOR_STORE_DIR)
                )
                self._vector_store = chroma_client.get_or_create_collection(
                    name="message_history",
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise RuntimeError(
                    "chromadb package not installed. Run: pip install chromadb"
                )
        return self._vector_store

    # ──────────────────────────────────────────────────────────────
    # Thread Context Assembly
    # ──────────────────────────────────────────────────────────────

    def _get_thread_context(self, msg: dict, user_history: list[dict]) -> str:
        """
        Get the last 3-5 messages from the same sender/group/business
        thread for a given user, to provide surrounding context.
        """
        conv_type = msg.get("conversation_type", "")
        thread_msgs = []

        for h in user_history:
            if h["message_id"] == msg["message_id"]:
                continue
            if conv_type == "group" and h.get("group_id") == msg.get("group_id"):
                thread_msgs.append(h)
            elif conv_type == "business" and h.get("business_id") == msg.get("business_id"):
                thread_msgs.append(h)
            elif conv_type == "personal" and h.get("sender_user_id") == msg.get("sender_user_id"):
                thread_msgs.append(h)

        # Take last 3 by timestamp (already sorted in CSV usually)
        thread_msgs = thread_msgs[-3:]

        if not thread_msgs:
            return "No prior messages in this thread."

        lines = []
        for t in thread_msgs:
            text_preview = (t.get("message_text", "") or "")[:100]
            lines.append(
                f"- [{t.get('created_at', '?')}] {t.get('sender_user_id', 'business')}: "
                f'"{text_preview}"'
            )
        return "\n".join(lines)

    def _get_event_summary(self, message_id: str) -> str:
        """Summarize the user's reaction to a historical message."""
        event = self.store.get_message_event(message_id)
        if not event:
            return "No reaction data available."

        parts = []
        if event.get("message_opened") == "1":
            parts.append("opened")
        if event.get("message_replied") == "1":
            parts.append("replied")
        if event.get("notification_dismissed") == "1":
            parts.append("dismissed notification")
        if event.get("muted_after_message") == "1":
            parts.append("muted sender after this message")
        if event.get("message_reported") == "1":
            parts.append("REPORTED this message")
        reaction_time = event.get("reaction_time_minutes", "")
        if reaction_time:
            parts.append(f"reacted in {reaction_time} minutes")

        return "User " + ", ".join(parts) + "." if parts else "No reaction recorded."

    # ──────────────────────────────────────────────────────────────
    # Main Indexing Pipeline
    # ──────────────────────────────────────────────────────────────

    def build_index(self):
        """
        Run the full contextual indexing pipeline over message_history.csv.

        For each historical message:
          1. Build thread context
          2. Get event summary
          3. Generate contextual header (local LLM)
          4. Create contextualized chunk
          5. Embed and store in vector DB
        """
        collection = self._get_vector_store()

        # Check what's already indexed
        existing_ids = set()
        try:
            existing = collection.get()
            existing_ids = set(existing["ids"]) if existing["ids"] else set()
        except Exception:
            pass

        # Group history by user for thread context lookup
        all_history = []
        for user_msgs in self.store.message_history.values():
            all_history.extend(user_msgs)

        total = len(all_history)
        new_count = 0

        print(f"  Indexing {total} historical messages...")

        for i, msg in enumerate(all_history):
            msg_id = msg["message_id"]

            # Skip already indexed
            if msg_id in existing_ids:
                continue

            user_id = msg["user_id"]
            user_history = self.store.message_history.get(user_id, [])

            # Step 1: Thread context
            thread_ctx = self._get_thread_context(msg, user_history)

            # Step 2: Event summary
            event_summary = self._get_event_summary(msg_id)

            # Step 3: Generate contextual header (use cache if available)
            if msg_id in self._chunks_cache:
                header = self._chunks_cache[msg_id]["header"]
            else:
                prompt = HEADER_PROMPT_TEMPLATE.format(
                    thread_context=thread_ctx,
                    event_summary=event_summary,
                    sender_id=msg.get("sender_user_id", msg.get("business_id", "unknown")),
                    conv_type=msg.get("conversation_type", ""),
                    entity_id=msg.get("group_id") or msg.get("business_id") or "personal",
                    timestamp=msg.get("created_at", ""),
                    message_text=(msg.get("message_text", "") or "")[:300],
                )
                header = self._generate_header(prompt)
                self._chunks_cache[msg_id] = {
                    "header": header,
                    "user_id": user_id,
                }

            # Step 4: Contextualized chunk
            raw_text = msg.get("message_text", "") or "[no text]"
            contextualized_chunk = f"{header}\n---\n{raw_text[:500]}"

            # Step 5: Embed and store
            embedding = self._embed_text(contextualized_chunk)

            metadata = {
                "user_id": user_id,
                "conversation_type": msg.get("conversation_type", ""),
                "group_id": msg.get("group_id", ""),
                "business_id": msg.get("business_id", ""),
                "sender_user_id": msg.get("sender_user_id", ""),
                "media_type": msg.get("media_type", ""),
                "forwarded_count": str(msg.get("forwarded_count", "0")),
                "created_at": msg.get("created_at", ""),
            }

            collection.add(
                ids=[msg_id],
                embeddings=[embedding],
                documents=[contextualized_chunk],
                metadatas=[metadata],
            )

            new_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"    [{i+1}/{total}] processed ({new_count} new)")

        # Save chunks cache
        self._save_chunks_cache()
        print(f"  ✅ Indexed {new_count} new chunks (total: {len(existing_ids) + new_count})")
