"""Configuration dataclass for the KG Conformity Agent."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConformityConfig:
    # Matching thresholds
    fuzzy_threshold:   float = 80.0   # rapidfuzz token_sort_ratio score (0–100)
    jaccard_threshold: float = 0.30   # property Jaccard similarity (0–1)

    # Guard against O(N²) explosion on very large KGs
    max_node_pairs: int = 10_000

    # LLM settings for the recommend node
    llm_model:       str   = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # Cap the number of candidates sent to the LLM prompt
    max_conformities_in_prompt: int = 60
