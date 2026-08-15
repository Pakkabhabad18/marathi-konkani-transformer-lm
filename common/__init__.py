"""Shared, language-agnostic utilities for the Phase 1 data pipelines.

Only *code* is shared between the Marathi and Konkani pipelines. Data,
tokenizers, vocabularies, splits, statistics and model weights are kept
completely separate per language, as the project specification requires.
See report/phase1_decisions.md, decision D-009.
"""
