"""
Message Notification Router — Main Orchestrator

Entry point for the full pipeline:
  1. Data Ingestion         — Load all CSVs
  2. Context Indexing       — Build contextual embeddings (one-time)
  3. Multimodal Normalizer  — OCR images, transcribe voice notes
  4. Context Assembly       — Build structured context per message
  5. Retrieval              — Semantic + structured evidence search
  6. Decision LLM           — Local LLM classification
  7. Output                 — Write output.csv

Usage:
    cd code
    python main.py
"""

import os
import sys
import time

# Ensure code/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR, OUTPUT_PATH
from data_loader import DataStore
from context_builder import build_user_context, build_message_context
from multimodal_normalizer import MultimodalNormalizer
from contextual_indexer import ContextualIndexer
from retriever import Retriever
from decision_llm import DecisionLLM
from output_writer import write_output, calibrate_confidence


def main():
    start_time = time.time()
    print("=" * 60)
    print("  Message Notification Router — Local Pipeline")
    print("=" * 60)

    # ── Stage 1: Data Ingestion ─────────────────────────────────
    print("\n[1/7] Loading dataset...")
    store = DataStore(DATASET_DIR)
    print(f"  [OK] Loaded {len(store.messages)} messages to route")
    print(f"  [OK] Loaded {len(store.users)} users, {len(store.groups)} groups, "
          f"{len(store.businesses)} businesses")

    # ── Stage 2: Pre-compute User Profiles ──────────────────────
    print("\n[2/7] Building user profiles...")
    user_contexts = {}
    for uid in store.users:
        user_contexts[uid] = build_user_context(uid, store)
    print(f"  [OK] Built {len(user_contexts)} user profiles")

    # ── Stage 3: Batch Media Pre-processing ────────────────────
    # Process all images and voice notes FIRST, before loading the LLM.
    # moondream (images) and faster-whisper (audio) run sequentially,
    # then Ollama unloads them to free VRAM for llama3.2.
    print("\n[3/8] Pre-processing media files...")
    normalizer = MultimodalNormalizer()

    # Collect all image and voice note items
    image_items = [
        (img_id, img["file_path"])
        for img_id, img in store.images.items()
    ]
    voice_items = [
        (vn_id, vn["file_path"])
        for vn_id, vn in store.voice_notes.items()
    ]

    if image_items:
        normalizer.batch_process_images(image_items)
    if voice_items:
        normalizer.batch_process_voice_notes(voice_items)

    # ── Stage 4: Contextual Indexing (one-time) ─────────────────
    print("\n[4/8] Building contextual index of message history...")
    indexer = ContextualIndexer(store)
    indexer.build_index()

    # ── Stage 5: Initialize Retriever & Decision LLM ────────────
    print("\n[5/8] Initializing retriever and decision LLM...")
    retriever = Retriever(store)
    decision_llm = DecisionLLM()

    # ── Stage 6-7: Process Each Message ─────────────────────────
    print(f"\n[6/8] Processing {len(store.messages)} messages...")
    predictions = []

    for i, msg in enumerate(store.messages):
        msg_id = msg["message_id"]
        print(f"\n  [{i+1}/{len(store.messages)}] {msg_id}", end="")

        # Build structured context
        ctx = build_message_context(msg, store, user_contexts)

        # Process media (if any)
        if ctx.media_type and ctx.media_id:
            if ctx.media_type == "image":
                file_path = store.get_image_path(ctx.media_id)
            elif ctx.media_type == "voice":
                file_path = store.get_voice_note_path(ctx.media_id)
            else:
                file_path = None

            if file_path:
                ctx.media_content = normalizer.normalize(
                    ctx.media_type, ctx.media_id, file_path
                )
                print(f" [{ctx.media_type}]", end="")

        # Retrieve evidence
        evidence_list, evidence_ids_str = retriever.retrieve(ctx)
        evidence_text = retriever.format_evidence_for_prompt(evidence_list)

        # LLM decision
        llm_result = decision_llm.classify(ctx, evidence_text)
        print(f" -> {llm_result['action']} ({llm_result['message_type']})", end="")

        # Record prediction
        predictions.append({
            "message_id": msg_id,
            "action": llm_result["action"],
            "message_type": llm_result["message_type"],
            "reason": llm_result["reason"],
            "confidence": calibrate_confidence(llm_result["confidence"]),
            "evidence_message_ids": evidence_ids_str,
        })

    # ── Stage 8: Write Output ───────────────────────────────────
    print(f"\n\n[7/8] Writing output...")
    write_output(predictions, str(OUTPUT_PATH))

    elapsed = time.time() - start_time
    print(f"\n[8/8] [DONE] Pipeline complete")
    print(f"  Wrote {len(predictions)} predictions to {OUTPUT_PATH}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("=" * 60)


if __name__ == "__main__":
    main()
