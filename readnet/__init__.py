"""Akshar: turn a child's reading into an ASER level.

Implements the post-ASR half of the Akshar / PadhAI blueprint — per-language
normalisation, weighted alignment with error classification, ASER placement,
forced alignment + GOP over any CTC model, and agreement metrics. See
README.md in this folder.
"""
