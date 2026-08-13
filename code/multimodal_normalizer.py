"""
Stage 2 — Multimodal Normalizer

Converts every message into a unified text representation:
  - Text messages: pass-through
  - Images: moondream via Ollama (OCR + visual captioning in one model)
  - Voice notes: ASR via faster-whisper

Results are cached to media_cache.json so re-runs skip processed files.

moondream handles both text extraction (OCR) and visual captioning natively.
Since it runs through Ollama, GPU loading/unloading is managed automatically.
"""

import base64
import json
import os
from pathlib import Path

from config import (
    OLLAMA_BASE_URL,
    MEDIA_CACHE_PATH,
    CACHE_DIR,
    IMAGES_DIR,
    AUDIO_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
)

# Model used for image understanding (OCR + captioning)
VISION_MODEL = "moondream"


class MultimodalNormalizer:
    """Normalize images and voice notes into text."""

    def __init__(self):
        self._cache = self._load_cache()
        self._whisper_model = None
        self._ollama_client = None

    # ──────────────────────────────────────────────────────────────
    # Ollama Client
    # ──────────────────────────────────────────────────────────────

    def _get_ollama_client(self):
        """Lazy-init Ollama client."""
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        return self._ollama_client

    # ──────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        """Load previously processed media results from disk."""
        if MEDIA_CACHE_PATH.exists():
            with open(MEDIA_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Persist cache to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MEDIA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def normalize(self, media_type: str, media_id: str, file_path: str) -> str:
        """
        Convert a media file to text.

        Args:
            media_type: "image" or "voice"
            media_id:   e.g. "img_001" or "vn_003"
            file_path:  relative path from images.csv / voice_notes.csv

        Returns:
            Extracted/transcribed text string.
        """
        # Check cache first
        if media_id in self._cache:
            return self._cache[media_id]

        if media_type == "image":
            result = self._process_image(media_id, file_path)
        elif media_type == "voice":
            result = self._process_voice(media_id, file_path)
        else:
            result = ""

        # Cache and persist
        self._cache[media_id] = result
        self._save_cache()
        return result

    def batch_process_images(self, image_items: list[tuple[str, str]]):
        """
        Pre-process all images in a batch before the main pipeline starts.
        This loads moondream once, processes all images, then Ollama can
        unload it to free VRAM for the LLM.

        Args:
            image_items: list of (media_id, file_path) tuples
        """
        uncached = [(mid, fp) for mid, fp in image_items if mid not in self._cache]
        if not uncached:
            print("  All images already cached, skipping.")
            return

        print(f"  Processing {len(uncached)} uncached images via moondream...")
        for i, (media_id, file_path) in enumerate(uncached):
            result = self._process_image(media_id, file_path)
            self._cache[media_id] = result
            print(f"    [{i+1}/{len(uncached)}] {media_id}: {result[:80]}...")

        self._save_cache()
        print(f"  ✅ All images processed and cached.")

    def batch_process_voice_notes(self, voice_items: list[tuple[str, str]]):
        """
        Pre-process all voice notes in a batch.
        Loads faster-whisper once and transcribes all audio files.

        Args:
            voice_items: list of (media_id, file_path) tuples
        """
        uncached = [(mid, fp) for mid, fp in voice_items if mid not in self._cache]
        if not uncached:
            print("  All voice notes already cached, skipping.")
            return

        print(f"  Transcribing {len(uncached)} uncached voice notes via faster-whisper...")
        for i, (media_id, file_path) in enumerate(uncached):
            result = self._process_voice(media_id, file_path)
            self._cache[media_id] = result
            print(f"    [{i+1}/{len(uncached)}] {media_id}: {result[:80]}...")

        self._save_cache()
        print(f"  ✅ All voice notes transcribed and cached.")

    # ──────────────────────────────────────────────────────────────
    # Image Processing via moondream (Ollama)
    # ──────────────────────────────────────────────────────────────

    def _process_image(self, media_id: str, file_path: str) -> str:
        """
        Analyze an image using moondream via Ollama.
        moondream handles both OCR (text extraction) and visual captioning.
        """
        abs_path = self._resolve_image_path(file_path, media_id)
        if not abs_path or not os.path.exists(abs_path):
            return f"[Image file not found: {file_path}]"

        temp_path = None
        try:
            # Convert image to RGB JPEG to ensure Ollama compatibility
            # (Ollama fails on WebP, RGBA PNGs, etc.)
            from PIL import Image
            import tempfile
            
            with Image.open(abs_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Save to a temporary JPEG
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                img.save(temp_path, format="JPEG")

            client = self._get_ollama_client()

            # moondream accepts images directly through Ollama's vision API
            response = client.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Analyze this WhatsApp image thoroughly. "
                        "First, extract ALL readable text from the image exactly as written. "
                        "Then describe what the image shows (poster, screenshot, receipt, meme, photo, etc). "
                        "If the image contains any links, phone numbers, or brand names, include them. "
                        "If the image appears to be a scam, phishing attempt, or contains suspicious content, say so."
                    ),
                    "images": [temp_path],
                }],
                options={"temperature": 0.0},
            )

            result = response["message"]["content"].strip()
            return f"[Image analysis]: {result}" if result else "[Image: no content extracted]"

        except Exception as e:
            # Fallback to pytesseract if moondream/Ollama fails
            print(f"    ⚠ moondream failed for {media_id}: {e}, falling back to pytesseract")
            return self._fallback_ocr(abs_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _fallback_ocr(self, image_path: str) -> str:
        """Fallback OCR using pytesseract when moondream is unavailable."""
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="eng")
            return f"[Image OCR]: {text.strip()}" if text.strip() else "[Image: no text extracted]"
        except ImportError:
            return "[OCR unavailable — install pytesseract and Pillow]"
        except Exception as e:
            return f"[OCR error: {e}]"

    def _resolve_image_path(self, file_path: str, media_id: str) -> str | None:
        """Resolve image path from CSV file_path field."""
        # Try the file_path from CSV directly
        if os.path.exists(file_path):
            return file_path
        # Try under dataset/media/images/
        candidate = IMAGES_DIR / f"{media_id}.jpg"
        if candidate.exists():
            return str(candidate)
        # Try relative to dataset
        candidate = Path(file_path)
        if candidate.exists():
            return str(candidate)
        return None

    # ──────────────────────────────────────────────────────────────
    # Voice Note Processing (ASR)
    # ──────────────────────────────────────────────────────────────

    def _process_voice(self, media_id: str, file_path: str) -> str:
        """Transcribe a voice note using faster-whisper."""
        abs_path = self._resolve_audio_path(file_path, media_id)
        if not abs_path or not os.path.exists(abs_path):
            return f"[Audio file not found: {file_path}]"

        return self._run_asr(abs_path)

    def _resolve_audio_path(self, file_path: str, media_id: str) -> str | None:
        """Resolve audio path from CSV file_path field."""
        if os.path.exists(file_path):
            return file_path
        candidate = AUDIO_DIR / f"{media_id}.mp3"
        if candidate.exists():
            return str(candidate)
        candidate = Path(file_path)
        if candidate.exists():
            return str(candidate)
        return None

    def _run_asr(self, audio_path: str) -> str:
        """
        Transcribe audio using faster-whisper.
        Falls back to OpenAI whisper if faster-whisper is not installed.
        """
        try:
            from faster_whisper import WhisperModel

            if self._whisper_model is None:
                # Auto-detect GPU: use CUDA if available, else CPU
                if WHISPER_DEVICE == "auto":
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                else:
                    device = WHISPER_DEVICE

                # Use float16 for GPU (faster), int8 for CPU (lower memory)
                compute_type = "float16" if device == "cuda" else "int8"

                self._whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=compute_type,
                )
                print(f"    Whisper loaded on {device} ({compute_type})")

            segments, _info = self._whisper_model.transcribe(
                audio_path, language=None  # Auto-detect language
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return f"[Voice transcript]: {text}" if text else "[Voice: no speech detected]"

        except ImportError:
            # Fallback to openai-whisper
            try:
                import whisper

                if self._whisper_model is None:
                    self._whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                result = self._whisper_model.transcribe(audio_path)
                text = result.get("text", "").strip()
                return f"[Voice transcript]: {text}" if text else "[Voice: no speech detected]"
            except ImportError:
                return "[ASR unavailable — install faster-whisper or openai-whisper]"
        except Exception as e:
            return f"[ASR error: {e}]"
