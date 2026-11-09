"""Deterministic train and test split: an item that falls in the test bucket never leaks into training."""
import hashlib

BUCKETS = 100
HEX_PREFIX_CHARS = 8
HEX_BASE = 16


def bucket(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    return int(digest[:HEX_PREFIX_CHARS], HEX_BASE) % BUCKETS


def is_test(item_id: str, percent: int) -> bool:
    return bucket(item_id) < percent
