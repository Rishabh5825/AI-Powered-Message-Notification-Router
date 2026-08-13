# Smart WhatsApp Message Notification Router — Architecture

## 1. Goal
Classify every incoming message (text / image / voice) into `notify`, `digest`, or `mute`, using message content **plus** personalization context (user behavior, group dynamics, business history, past interactions). Output: `message_id, action, message_type, reason, confidence, evidence_message_ids`.

Hard constraint: **no paid API calls**. Everything (OCR, ASR, embeddings, LLM reasoning) runs on local, open-source models.

---

## 2. High-Level Pipeline

```
messages.csv ──┐
images.csv ────┼──▶ [Multimodal Normalizer] ──▶ unified_text
voice_notes.csv┘                                    │
                                                      ▼
                                        [Contextual Chunk + Header Gen] (local LLM)
                                                      │
                                                      ▼
                                        [Local Embedding Model] ──▶ [Vector Store]
                                                      │
users/groups/business/history CSVs ──▶ [Context Builder] (structured, no embedding needed)
                                                      │
                                                      ▼
                                   [Retriever] (contextual-embedding search + structured filters)
                                                      │
                                                      ▼
                                   [Local Decision LLM] ──▶ action, type, reason, confidence, evidence
                                                      │
                                                      ▼
                                              output.csv
```

---

## 3. Multimodal Ingestion (answering Q2)

Every message is normalized to a text representation before it enters the RAG pipeline.

| Input type | Model | Why |
|---|---|---|
| `message_text` | pass-through | already text |
| `image` (media_type=image) | **Baidu Unlimited-OCR** (`baidu/Unlimited-OCR`, MIT license, open-sourced June 2026), run locally via Transformers/vLLM/SGLang | 3B-class MoE vision-language OCR model built on DeepSeek-OCR, using Reference Sliding-Window Attention to hold the KV cache flat — reads multi-image/multi-page content (long forwarded circulars, chained screenshots) in a single pass instead of stitching page-by-page OCR. SOTA on OmniDocBench, MIT-licensed, fully offline, runs on a single 12GB+ GPU |
| `image` (non-document, e.g. product photo, meme with little/no text) | Unlimited-OCR still extracts any embedded text; if output is empty/sparse, fall back to a small local VLM (e.g. `moondream2`, ~1.9B, CPU-runnable) for a one-line visual caption | Gives the router something to reason over even when the image has no readable text |
| `voice` (media_type=voice) | **faster-whisper** (`small` or `medium` model, CTranslate2 backend) run locally | Best accuracy/speed/offline tradeoff for Hindi/English mixed voice notes on commodity hardware |

> Note: an earlier version of this doc assumed "Unlimited OCR" didn't exist — that was incorrect. Baidu released it June 22, 2026 (arXiv 2606.23050), after this system's original knowledge cutoff. It's real, open-source, and a better fit here than PaddleOCR for multi-image forwarded documents.

Output of this stage: a single `unified_text` field per message (OCR text / caption / transcript, whichever applies), stored alongside the original row.

---

## 4. Contextual Retrieval RAG (answering Q1)

Standard RAG chunks (e.g. raw rows from `message_history.csv`) are often too short/ambiguous in isolation ("ok will do", "same as last time"). We fix this with **contextual retrieval**, run once, offline, at indexing time:

**Step-by-step:**
1. For each historical message (from `message_history.csv` + its `message_events.csv` outcome), assemble the *raw chunk*: sender, timestamp, text/OCR/transcript, and the event outcome (opened/replied/dismissed/muted/reported).
2. Feed the raw chunk **and** its surrounding context (same user + same sender/group thread, last 3–5 messages) to the **local LLM** with the prompt:
   > "What context does a reader need to understand this chunk without seeing the rest of the document? Answer in 2–3 sentences."
3. Prepend the generated header to the raw chunk → `contextualized_chunk = header + "\n" + raw_chunk`.
4. Embed `contextualized_chunk` with a local embedding model and store in a local vector index, tagged with `user_id`, `sender_id/business_id`, `group_id`, `message_type`, `outcome`.

This is done **once as a batch preprocessing job** over `message_history.csv` (not per incoming message), so the cost is paid up front, not at inference time.

**Example:**
- Raw chunk: `"Reply STOP to unsubscribe" — user dismissed`
- Generated header: *"This is a promotional message from business_094 sent to u_007 on a previous date. The user dismissed it without opening, part of a recurring pattern of ignoring this sender's offers."*

---

## 5. Retrieval at Inference Time

For each incoming message:
1. **Structured filter** (cheap, no LLM/embedding needed): pull all rows from `users.csv`, `groups.csv`, `group_members.csv`, `business_accounts.csv`, `user_business_history.csv`, `daily_notification_summary.csv` matching this `user_id` / `group_id` / `business_id` / `sender_user_id`.
2. **Semantic search**: embed the incoming message's `unified_text` (with a lightweight auto-generated context header, same technique as §4) using the same local embedding model, and query the vector store filtered to this `user_id` + this `sender_id/business_id/group_id` for top-k similar past messages.
3. Combine (1) + (2) into a single structured context block passed to the decision LLM.

---

## 6. Local Model Stack (answering Q3 — no API calls)

All inference runs on-device via **Ollama** (simplest local model server, single binary, OpenAI-compatible local endpoint at `localhost:11434`, zero cost, zero external calls).

| Task | Local model | Notes |
|---|---|---|
| Contextual header generation (§4) | `qwen2.5:7b-instruct` (or `llama3.1:8b-instruct` if more RAM available) via Ollama | Good instruction-following at small size; runs on CPU or modest GPU |
| Embeddings | `nomic-embed-text` or `bge-m3` via Ollama | Both run locally, `bge-m3` has strong multilingual (Hindi/English) support |
| Final decision (action/type/reason/confidence) | Same `qwen2.5:7b-instruct` (or `14b` if hardware allows, for better reasoning) | Structured JSON output mode enforced via prompt + JSON schema validation |
| OCR | Baidu Unlimited-OCR (local, GPU, 12GB+ VRAM recommended) | See §3 |
| ASR | faster-whisper `small`/`medium` (local, CPU/GPU) | See §3 |
| Vector store | **ChromaDB** (local, embedded, file-based) or **FAISS** if a lighter dependency is preferred | No server needed, persists to disk |

Everything above runs fully offline after the one-time model download (`ollama pull qwen2.5:7b-instruct`, etc.) — no per-call API cost, no external network dependency at runtime.

---

## 7. Decision Stage

The decision LLM receives, per message:
- `unified_text` (+ contextual header)
- Sender/business/group structured facts (verified?, opted-in?, admin role?, recent orders?, quiet hours?, group size/read-reply rate?)
- Top-k retrieved historical evidence (with their outcomes: opened/muted/reported)
- Daily notification load for the user (to avoid over-notifying)

It is prompted to return **strict JSON**:
```json
{
  "action": "notify | digest | mute",
  "message_type": "urgent | personal | event | promotion | business_update | greeting | forward | scam | spam | unknown",
  "reason": "short human-readable sentence",
  "confidence": 0.0-1.0,
  "evidence_message_ids": ["message_00xx", ...] or "none"
}
```
Deterministic decoding (temperature=0) + JSON schema validation with a retry-on-malformed-output loop.

---

## 8. Output
Results are written to `output.csv` matching `dataset/output.csv`'s template columns:
`message_id, action, message_type, reason, confidence, evidence_message_ids`

---

## 9. Why This Design
- **Contextual retrieval** fixes the "meaningless chunk" problem by giving every historical chunk a self-contained header before embedding, so semantic search actually retrieves *relevant precedent*, not just lexically similar noise.
- **Fully local stack** (Ollama + PaddleOCR + faster-whisper + Chroma) satisfies the no-API-cost constraint end-to-end.
- **Structured + semantic hybrid retrieval** ensures hard facts (opt-outs, verification, quiet hours) are never left to the LLM's judgment alone — they're injected directly, reducing hallucinated reasoning.
