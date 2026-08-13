"""
Stage 1 — Data Ingestion

Loads all CSV files from dataset/ into in-memory dictionaries.
Builds lookup indexes for O(1) access by primary and composite keys.

Handles multiline CSV fields (message_text with embedded newlines)
via Python's standard csv.DictReader.
"""

import csv
from pathlib import Path
from config import DATASET_DIR


class DataStore:
    """Central in-memory store for all dataset files."""

    def __init__(self, dataset_dir: str | Path = DATASET_DIR):
        self.base = Path(dataset_dir)

        # ── Primary tables (keyed by ID) ──────────────────────────
        self.messages = self._load("messages.csv")
        self.sample_messages = self._load("sample_messages.csv")
        self.users = self._load_keyed("users.csv", "user_id")
        self.groups = self._load_keyed("groups.csv", "group_id")
        self.businesses = self._load_keyed("business_accounts.csv", "business_id")
        self.images = self._load_keyed("images.csv", "image_id")
        self.voice_notes = self._load_keyed("voice_notes.csv", "voice_note_id")

        # ── Relationship tables (grouped by key) ─────────────────
        self.group_members = self._load_grouped("group_members.csv", "group_id")
        self.group_members_by_user = self._load_grouped("group_members.csv", "user_id")
        self.user_business_history = self._load_grouped(
            "user_business_history.csv", "user_id"
        )
        self.user_business_history_by_pair = self._load_grouped(
            "user_business_history.csv", ("user_id", "business_id")
        )
        self.message_history = self._load_grouped("message_history.csv", "user_id")
        self.message_events = self._load_grouped("message_events.csv", "user_id")
        self.message_events_by_msg = self._load_keyed("message_events.csv", "message_id")
        self.daily_summary = self._load_grouped(
            "daily_notification_summary.csv", "user_id"
        )

        # ── Output template ───────────────────────────────────────
        self.output_template = self._load("output.csv")

    # ──────────────────────────────────────────────────────────────
    # Internal loaders
    # ──────────────────────────────────────────────────────────────

    def _load(self, filename: str) -> list[dict]:
        """Load a CSV into a list of dicts."""
        filepath = self.base / filename
        with open(filepath, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_keyed(self, filename: str, key: str) -> dict[str, dict]:
        """Load a CSV and index rows by a unique key column."""
        return {row[key]: row for row in self._load(filename)}

    def _load_grouped(
        self, filename: str, key: str | tuple
    ) -> dict[str | tuple, list[dict]]:
        """
        Load a CSV and group rows by a single key or composite key tuple.

        Example:
            _load_grouped("group_members.csv", "group_id")
              → {"group_001": [row, row, ...], ...}

            _load_grouped("user_business_history.csv", ("user_id", "business_id"))
              → {("u_001", "business_001"): [row, ...], ...}
        """
        data = self._load(filename)
        result: dict = {}
        if isinstance(key, tuple):
            for row in data:
                k = tuple(row[k] for k in key)
                result.setdefault(k, []).append(row)
        else:
            for row in data:
                result.setdefault(row[key], []).append(row)
        return result

    def get_user(self, user_id: str) -> dict:
        """Get user profile or empty dict if missing."""
        return self.users.get(user_id, {})

    def get_group(self, group_id: str) -> dict:
        """Get group metadata or empty dict if missing."""
        return self.groups.get(group_id, {})

    def get_business(self, business_id: str) -> dict:
        """Get business account info or empty dict if missing."""
        return self.businesses.get(business_id, {})

    def get_image_path(self, media_id: str) -> str | None:
        """Resolve image_id → file_path from images.csv."""
        img = self.images.get(media_id)
        return img["file_path"] if img else None

    def get_voice_note_path(self, media_id: str) -> str | None:
        """Resolve voice_note_id → file_path from voice_notes.csv."""
        vn = self.voice_notes.get(media_id)
        return vn["file_path"] if vn else None

    def get_user_group_membership(self, user_id: str, group_id: str) -> dict | None:
        """Get a specific user's membership record in a group."""
        memberships = self.group_members.get(group_id, [])
        for m in memberships:
            if m["user_id"] == user_id:
                return m
        return None

    def get_user_business_relation(self, user_id: str, business_id: str) -> dict | None:
        """Get a user's history with a specific business."""
        rows = self.user_business_history_by_pair.get((user_id, business_id), [])
        return rows[0] if rows else None

    def get_message_event(self, message_id: str) -> dict | None:
        """Get the event record for a historical message."""
        return self.message_events_by_msg.get(message_id)
