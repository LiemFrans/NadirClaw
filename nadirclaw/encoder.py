"""Shared SentenceTransformer singleton for NadirClaw."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_shared_encoder: SentenceTransformer | None = None


def get_shared_encoder_sync() -> SentenceTransformer:
    """
    Lazily initialize and return a shared SentenceTransformer instance.
    The first call loads the model (~80 MB download on first run).
    """
    global _shared_encoder
    if _shared_encoder is None:
        logger.info("Loading SentenceTransformer encoder: all-MiniLM-L6-v2")
        _shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer encoder loaded")
    return _shared_encoder
