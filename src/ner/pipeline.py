"""
NER pipeline with entity normalization.

Uses dslim/bert-base-NER (fine-tuned on CoNLL-2003) via HuggingFace pipeline.
After extracting entity spans, applies brand normalization to map surface forms
(e.g. "Apple", "$AAPL", "apple inc") to canonical brand IDs.

Why a separate NER module (not a 4th head on the multi-task model):
    NER produces character-level span outputs (start, end, entity_group),
    not a single class label per document. Adding a span-detection head to
    the multi-task model would require token-level labels aligned with the
    tokeniser subwords — a different architecture and training objective.
    In production, NER typically runs as a separate lightweight service
    at lower throughput than the main classifier.
"""

from __future__ import annotations

from typing import Optional

from transformers import pipeline as hf_pipeline

from configs.config import NER_MODEL_ID, BRAND_NORMALIZATION, NER_ENTITY_TYPES


class NERPipeline:
    """
    Named Entity Recognition with brand normalization.

    Wraps HuggingFace's token classification pipeline and adds
    post-processing: entity merging, normalization, and canonical ID lookup.
    """

    def __init__(
        self,
        model_id: str = NER_MODEL_ID,
        device: int = -1,  # -1 = CPU, 0 = first GPU
        batch_size: int = 16,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self._pipeline = hf_pipeline(
            "ner",
            model=model_id,
            aggregation_strategy="simple",  # merge subword tokens into word spans
            device=device,
            batch_size=batch_size,
        )

    def extract(self, text: str) -> list[dict]:
        """
        Extract and normalize named entities from a single text.

        Returns list of dicts:
            {
                "text":         "Apple",
                "entity_type":  "ORG",
                "score":        0.998,
                "start":        12,
                "end":          17,
                "canonical_id": "brand:apple" | None
            }
        """
        if not text or not text.strip():
            return []

        raw_entities = self._pipeline(text[:512])  # truncate for safety

        results = []
        for ent in raw_entities:
            entity_type = ent.get("entity_group", "").upper()
            if entity_type not in NER_ENTITY_TYPES:
                continue

            surface = ent.get("word", "").strip()
            canonical_id = _normalize_entity(surface, entity_type)

            results.append(
                {
                    "text": surface,
                    "entity_type": entity_type,
                    "score": round(float(ent.get("score", 0)), 4),
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                    "canonical_id": canonical_id,
                }
            )

        return results

    def extract_batch(self, texts: list[str]) -> list[list[dict]]:
        """
        Extract entities from a batch of texts.
        Uses HuggingFace pipeline's built-in batching for efficiency.
        """
        if not texts:
            return []

        # Truncate and handle empty texts
        safe_texts = [t[:512] if t and t.strip() else " " for t in texts]
        batch_results = self._pipeline(safe_texts)

        all_entities = []
        for raw_entities in batch_results:
            entities = []
            for ent in raw_entities:
                entity_type = ent.get("entity_group", "").upper()
                if entity_type not in NER_ENTITY_TYPES:
                    continue
                surface = ent.get("word", "").strip()
                canonical_id = _normalize_entity(surface, entity_type)
                entities.append(
                    {
                        "text": surface,
                        "entity_type": entity_type,
                        "score": round(float(ent.get("score", 0)), 4),
                        "start": int(ent.get("start", 0)),
                        "end": int(ent.get("end", 0)),
                        "canonical_id": canonical_id,
                    }
                )
            all_entities.append(entities)

        return all_entities

    def get_brands(self, entities: list[dict]) -> list[str]:
        """Return list of canonical brand IDs mentioned in an entity list."""
        return list(
            {
                e["canonical_id"]
                for e in entities
                if e["canonical_id"] and e["canonical_id"].startswith("brand:")
            }
        )


def _normalize_entity(surface: str, entity_type: str) -> Optional[str]:
    """
    Map a surface entity mention to its canonical brand ID.

    Normalisation steps:
    1. Lowercase and strip leading $, #, @
    2. Look up in the BRAND_NORMALIZATION table
    3. Return canonical ID or None if not a known brand
    """
    if not surface:
        return None

    # Only normalise ORG and MISC entities (not PER or LOC)
    if entity_type not in ("ORG", "MISC"):
        return None

    key = surface.lower().strip().lstrip("$#@").strip()

    # Direct lookup
    if key in BRAND_NORMALIZATION:
        return BRAND_NORMALIZATION[key]

    # Partial match: check if any known brand name is a substring
    for brand_key, brand_id in BRAND_NORMALIZATION.items():
        if brand_key in key or key in brand_key:
            return brand_id

    return None
