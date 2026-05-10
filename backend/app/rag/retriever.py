from collections import defaultdict
from typing import Any

from rapidfuzz import fuzz

from app.rag.catalog_store import CatalogStore


TECH_KEYWORDS = {"java", "developer", "backend", "engineer", "api", "microservice", "spring"}
CLIENT_FACING_KEYWORDS = {"client", "stakeholder", "communication", "present", "customer"}
BEHAVIOR_KEYWORDS = {"personality", "behavior", "situational", "judgment"}
COGNITIVE_KEYWORDS = {"cognitive", "reasoning", "analytical", "problem solving", "ability", "learning", "adapt", "intelligence"}


class HybridRetriever:
    def __init__(self, store: CatalogStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        semantic_hits = self.store.search_vectors(query, top_k=30)
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, dict[str, Any]] = {}

        q_lower = query.lower()
        wants_tech = any(token in q_lower for token in TECH_KEYWORDS)
        wants_client_facing = any(token in q_lower for token in CLIENT_FACING_KEYWORDS)
        wants_behavior = any(token in q_lower for token in BEHAVIOR_KEYWORDS)
        wants_cognitive = any(token in q_lower for token in COGNITIVE_KEYWORDS)

        for score, doc in semantic_hits:
            key = doc["url"]
            scores[key] += score * 0.75
            docs[key] = doc

            text_blob = " ".join(
                [
                    doc.get("name", ""),
                    doc.get("description", ""),
                    " ".join(doc.get("skills", [])),
                    " ".join(doc.get("job_roles", [])),
                    " ".join(doc.get("tags", [])),
                ]
            ).lower()
            token_score = fuzz.partial_ratio(q_lower, text_blob) / 100.0
            scores[key] += token_score * 0.25
            scores[key] += self._domain_boost(
                query=q_lower,
                text_blob=text_blob,
                wants_tech=wants_tech,
                wants_client_facing=wants_client_facing,
                wants_behavior=wants_behavior,
                wants_cognitive=wants_cognitive,
            )

        ranked = sorted(docs.values(), key=lambda d: scores[d["url"]], reverse=True)
        return ranked[: max(1, min(top_k, 10))]

    def _domain_boost(
        self,
        query: str,
        text_blob: str,
        wants_tech: bool,
        wants_client_facing: bool,
        wants_behavior: bool,
        wants_cognitive: bool,
    ) -> float:
        bonus = 0.0

        if wants_tech:
            if any(term in text_blob for term in {"java", ".net", "ado.net", "framework", "developer"}):
                bonus += 0.35
            if any(term in text_blob for term in {"accounts payable", "accounts receivable"}):
                bonus -= 0.40

        if wants_client_facing:
            if any(term in text_blob for term in {"situational", "judgment", "language", "personality", "communication"}):
                bonus += 0.25

        if wants_behavior:
            if any(term in text_blob for term in {"opq", "motivational", "situational", "personality"}):
                bonus += 0.25

        if wants_cognitive:
            if any(term in text_blob for term in {"verify", "cognitive", "reasoning", "analytical", "problem solving", "ability", "learning", "adapt", "intelligence"}):
                bonus += 0.45
            elif any(term in text_blob for term in {"developer", "engineer", ".net", "java", "application", "framework"}):
                bonus -= 0.15

        if "java" in query and "java" not in text_blob and ".net" not in text_blob:
            bonus -= 0.10

        return bonus

    def exact_name_lookup(self, name: str) -> dict[str, Any] | None:
        choices = self.store.metadata
        best = None
        best_score = 0.0
        for item in choices:
            score = fuzz.ratio(name.lower(), item["name"].lower())
            if score > best_score:
                best_score = score
                best = item
        return best if best and best_score >= 70 else None
