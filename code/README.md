# Message Notification Router (Local Pipeline)

This is a fully local, privacy-first, GPU-accelerated implementation of the HackerRank Orchestrate Message Notification Router challenge. It evaluates multimodal WhatsApp messages (Text, Images, Voice Notes) and assigns routing actions (`notify`, `digest`, `mute`) based on user history and context.

## 🌟 Key Architectural Features

This architecture completely bypasses paid APIs, strictly adhering to a **zero-cost, open-source AI stack**:

- **Decision Engine (LLM):** `llama3.2:3b-instruct` running via Ollama. It evaluates a heavily structured, sandboxed prompt injected with user risk profiles and RAG evidence. Fits perfectly in 4GB VRAM.
- **Vision & OCR:** `moondream` (via Ollama). Unifies text extraction and visual captioning into a single lightweight VLM. 
- **Voice Transcription (ASR):** `faster-whisper` (CTranslate2 backend). GPU-accelerated inference.
- **Historical Evidence (RAG):** `ChromaDB` + `nomic-embed-text` (Ollama). Contextually indexes historical messages for semantic retrieval.
- **VRAM Management:** Employs a **Sequential Stage Processing** architecture. Heavy multimodal models (`moondream`, `faster-whisper`) are loaded in batch stages, cached to disk, and unloaded from GPU memory before the main LLM (`llama3.2`) pipeline spins up, ensuring smooth execution even on a 4GB GTX 1650.

---

## 🛠️ Setup & Installation

### 1. Install System Dependencies
- Install **Python 3.11+**
- Install **Ollama** from [ollama.com](https://ollama.com)

### 2. Install Python Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate     # Windows
source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

### 3. Pull Required Local Models
Before running the pipeline, ensure the Ollama service is running and pull the required models:
```bash
ollama pull llama3.2
ollama pull moondream
ollama pull nomic-embed-text
```

---

## 🚀 Running the Pipeline

### 1. The Main Orchestrator
To execute the pipeline and generate `output.csv`:
```bash
# Ensure you are inside the code directory
cd code
python main.py
```
*Note: The first run will take longer as it builds the ChromaDB embeddings and caches media processing results. Subsequent runs will instantly load media from `dataset/.cache/media_cache.json`.*

### 2. Self-Evaluation
To benchmark your predictions against the 53 sample ground-truth messages:
```bash
python evaluation/main.py
```
This script will output your overall Accuracy %, Action matching, Message Type matching, and detailed mismatch diagnostics for debugging LLM logic.

---

## 📁 Project Structure

```text
code/
├── config.py                  # Global project settings, model definitions, and paths
├── data_loader.py             # Stage 1: Fast CSV ingestion and indexed table lookups
├── context_builder.py         # Stage 2: Assembles User/Group/Business risk profiles
├── multimodal_normalizer.py   # Stage 3: OCR (moondream/tesseract) + ASR (whisper) batch caching
├── contextual_indexer.py      # Stage 4: One-time ChromaDB vector indexing of historical messages
├── retriever.py               # Stage 5: Hybrid semantic & structured historical evidence retrieval
├── decision_llm.py            # Stage 6: Sandboxed prompt execution via Ollama JSON mode
├── output_writer.py           # Stage 7: output.csv generation and confidence calibration
├── main.py                    # Orchestrator tying all stages together
└── evaluation/
    └── main.py                # Local testing and diagnostic script
```
