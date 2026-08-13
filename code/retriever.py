"""
Stage 4 — Retriever

Hybrid retrieval combining:
  1. Structured filter:  Hard facts from CSVs (already in MessageContext)
  2. Semantic search:    Vector similarity against contextual-embedded history

Returns top-k evidence message IDs for the decision LLM.
"""

from config import (
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    VECTOR_STORE_DIR,
    TOP_K_EVIDENCE,
)
from context_builder import MessageContext
from data_loader import DataStore


class Retriever:
    """Hybrid structured + semantic evidence retriever."""

    def __init__(self, store: DataStore):
        self.store = store
        self._ollama_client = None
        self._vector_store = None

    # ──────────────────────────────────────────────────────────────
    # Clients
    # ──────────────────────────────────────────────────────────────

    def _get_ollama_client(self):
        """Lazy-init Ollama client."""
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        return self._ollama_client

    def _get_vector_store(self):
        """Lazy-init ChromaDB collection."""
        if self._vector_store is None:
            import chromadb

            chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            self._vector_store = chroma_client.get_or_create_collection(
                name="message_history",
                metadata={"hnsw:space": "cosine"},
            )
        return self._vector_store

    def _embed_text(self, text: str) -> list[float]:
        """Embed a query string using the local model."""
        client = self._get_ollama_client()
        response = client.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]

    # ──────────────────────────────────────────────────────────────
    # Semantic Search
    # ──────────────────────────────────────────────────────────────

    def _semantic_search(self, ctx: MessageContext, top_k: int = TOP_K_EVIDENCE) -> list[dict]:
        """
        Embed the incoming message and query the vector store,
        filtered to this user + sender/group/business scope.
        """
        collection = self._get_vector_store()

        # Build the query text from unified message content
        query_parts = []
        if ctx.message_text:
            query_parts.append(ctx.message_text[:300])
        if ctx.media_content:
            query_parts.append(ctx.media_content[:200])
        query_text = " ".join(query_parts) if query_parts else "general message"

        # Build metadata filter scoped to this user + conversation entity
        where_filter = {"user_id": ctx.user_id}

        # Narrow by conversation entity for more relevant results
        if ctx.conversation_type == "group" and ctx.group_id:
            where_filter = {
                "$and": [
                    {"user_id": ctx.user_id},
                    {"group_id": ctx.group_id},
                ]
            }
        elif ctx.conversation_type == "business" and ctx.business_id:
            where_filter = {
                "$and": [
                    {"user_id": ctx.user_id},
                    {"business_id": ctx.business_id},
                ]
            }
        elif ctx.conversation_type == "personal" and ctx.sender_user_id:
            where_filter = {
                "$and": [
                    {"user_id": ctx.user_id},
                    {"sender_user_id": ctx.sender_user_id},
                ]
            }

        # Embed query and search
        query_embedding = self._embed_text(query_text)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # If filtered search returns nothing, try broader user-only filter
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    where={"user_id": ctx.user_id},
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        # Parse results
        evidence = []
        if results and results["ids"] and results["ids"][0]:
            for i, msg_id in enumerate(results["ids"][0]):
                evidence.append({
                    "message_id": msg_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 1.0,
                })

        return evidence

    # ──────────────────────────────────────────────────────────────
    # Evidence Enrichment
    # ──────────────────────────────────────────────────────────────

    def _enrich_with_events(self, evidence: list[dict]) -> list[dict]:
        """Add user reaction data (opened/dismissed/reported) to each evidence item."""
        for item in evidence:
            event = self.store.get_message_event(item["message_id"])
            if event:
                item["event"] = {
                    "opened": event.get("message_opened", "0"),
                    "replied": event.get("message_replied", "0"),
                    "dismissed": event.get("notification_dismissed", "0"),
                    "reported": event.get("message_reported", "0"),
                    "muted_after": event.get("muted_after_message", "0"),
                }
            else:
                item["event"] = None
        return evidence

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def retrieve(self, ctx: MessageContext) -> tuple[list[dict], str]:
        """
        Run hybrid retrieval for a message.

        Returns:
            evidence_list: List of evidence dicts with message_id, document, event data
            evidence_ids_str: Semicolon-separated string of evidence IDs for output.csv
        """
        # Semantic search over contextual embeddings
        evidence = self._semantic_search(ctx)

        # Enrich with event data
        evidence = self._enrich_with_events(evidence)

        # Build the output string
        if evidence:
            evidence_ids = [e["message_id"] for e in evidence]
            evidence_ids_str = ";".join(evidence_ids)
        else:
            evidence_ids_str = "none"

        return evidence, evidence_ids_str

    def format_evidence_for_prompt(self, evidence: list[dict]) -> str:
        """Format evidence items into a readable string for the decision LLM prompt."""
        if not evidence:
            return "No relevant historical messages found."

        lines = []
        for e in evidence:
            doc_preview = (e.get("document", "") or "")[:150]
            event_str = ""
            if e.get("event"):
                ev = e["event"]
                reactions = []
                if ev["opened"] == "1":
                    reactions.append("opened")
                if ev["replied"] == "1":
                    reactions.append("replied")
                if ev["dismissed"] == "1":
                    reactions.append("dismissed")
                if ev["reported"] == "1":
                    reactions.append("REPORTED")
                if ev["muted_after"] == "1":
                    reactions.append("muted_after")
                event_str = f" | User: {', '.join(reactions)}" if reactions else ""

            lines.append(f"- {e['message_id']}: \"{doc_preview}\"{event_str}")

        return "\n".join(lines)
