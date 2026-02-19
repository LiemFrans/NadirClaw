"""Shared SentenceTransformer singleton for NadirClaw."""

import logging
from threading import Lock

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_shared_encoder: SentenceTransformer | None = None
_encoder_lock = Lock()


def get_shared_encoder_sync() -> SentenceTransformer:
    """
    Lazily initialize and return a shared SentenceTransformer instance.
    The first call loads the model (~80 MB download on first run).
    Uses double-checked locking to avoid redundant loads.
    """
    global _shared_encoder
    if _shared_encoder is None:
        with _encoder_lock:
            if _shared_encoder is None:
                logger.info("Loading SentenceTransformer encoder: all-MiniLM-L6-v2")
                _shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer encoder loaded")
    return _shared_encoder
