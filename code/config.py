"""
Configuration constants for the Message Notification Router.

All paths, model names, thresholds, and pipeline settings live here.
Adjust OLLAMA_MODEL / EMBED_MODEL based on your hardware.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = PROJECT_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"
OUTPUT_PATH = DATASET_DIR / "output.csv"

# Cache files (auto-generated, avoid re-processing media on re-runs)
CACHE_DIR = PROJECT_ROOT / "code" / ".cache"
MEDIA_CACHE_PATH = CACHE_DIR / "media_cache.json"
VECTOR_STORE_DIR = CACHE_DIR / "vector_store"
CONTEXTUAL_CHUNKS_PATH = CACHE_DIR / "contextual_chunks.json"

# ---------------------------------------------------------------------------
# Ollama — Local LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"       # Decision + contextual header LLM (fits in 4GB VRAM)
OLLAMA_TEMPERATURE = 0.0                     # Deterministic decoding

# ---------------------------------------------------------------------------
# Embedding Model (via Ollama)
# ---------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"             # Or "bge-m3" for stronger multilingual

# ---------------------------------------------------------------------------
# Vision Model (via Ollama) — used for image OCR + captioning
# ---------------------------------------------------------------------------
VISION_MODEL = "moondream"                   # ~1.7B, handles OCR + captions

# ASR: faster-whisper
WHISPER_MODEL_SIZE = "small"                 # "small" or "medium"
WHISPER_DEVICE = "auto"                      # "auto", "cpu", or "cuda"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_STORE_BACKEND = "chromadb"             # "chromadb" or "faiss"
TOP_K_EVIDENCE = 5                            # Number of evidence chunks to retrieve
CONTEXT_HEADER_MAX_TOKENS = 150               # Max tokens for contextual header

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment",
    "business_update", "promotion", "greeting",
    "forward", "spam", "scam", "unknown",
}
