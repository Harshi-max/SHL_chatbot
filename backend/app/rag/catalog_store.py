import json
import pickle
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.schemas import AssessmentRecord
from app.utils.logging import get_logger


logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog.index"
META_PATH = DATA_DIR / "catalog_meta.pkl"


def _norm(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec / denom


class CatalogStore:
    _instance: "CatalogStore | None" = None

    def __new__(cls) -> "CatalogStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.records: list[AssessmentRecord] = []
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self.model: SentenceTransformer | None = None

    @property
    def count(self) -> int:
        return len(self.records)

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model

    def load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CATALOG_PATH.exists():
            CATALOG_PATH.write_text("[]", encoding="utf-8")
            logger.warning("Created empty catalog at %s", CATALOG_PATH)

        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.records = [AssessmentRecord.model_validate(item) for item in raw]
        logger.info("Loaded %d catalog records", len(self.records))

        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with META_PATH.open("rb") as fh:
                self.metadata = pickle.load(fh)
            # Fallback safety: if catalog is empty but index metadata exists,
            # use metadata as source of truth to avoid zero-result regressions.
            if not self.records and self.metadata:
                self.records = [AssessmentRecord.model_validate(item) for item in self.metadata]
                logger.warning(
                    "Catalog JSON empty; recovered %d records from FAISS metadata",
                    len(self.records),
                )
            return

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.records:
            self.index = None
            self.metadata = []
            return

        model = self._get_model()
        docs = [
            " | ".join(
                [
                    r.name,
                    r.description,
                    ",".join(r.skills),
                    ",".join(r.job_roles),
                    ",".join(r.tags),
                ]
            )
            for r in self.records
        ]
        embeddings = model.encode(docs, normalize_embeddings=True)
        vectors = np.asarray(embeddings, dtype="float32")
        vectors = _norm(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        self.metadata = [r.model_dump(mode="json") for r in self.records]
        faiss.write_index(index, str(INDEX_PATH))
        with META_PATH.open("wb") as fh:
            pickle.dump(self.metadata, fh)

    def search_vectors(self, query: str, top_k: int = 20) -> list[tuple[float, dict[str, Any]]]:
        if not self.index or not self.records:
            return []
        model = self._get_model()
        q = model.encode([query], normalize_embeddings=True)
        qv = np.asarray(q, dtype="float32")
        qv = _norm(qv)
        scores, ids = self.index.search(qv, min(top_k, len(self.records)))
        out: list[tuple[float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            out.append((float(score), self.metadata[idx]))
        return out
