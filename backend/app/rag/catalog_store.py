import json
import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import AssessmentRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
TFIDF_MODEL_PATH = DATA_DIR / "tfidf_model.pkl"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.pkl"
META_PATH = DATA_DIR / "catalog_meta.pkl"

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
        self.metadata: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None

    @property
    def count(self) -> int:
        return len(self.records)

    def load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CATALOG_PATH.exists():
            CATALOG_PATH.write_text("[]", encoding="utf-8")
            logger.warning("Created empty catalog at %s", CATALOG_PATH)

        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.records = [AssessmentRecord.model_validate(item) for item in raw]
        logger.info("Loaded %d catalog records", len(self.records))

        if TFIDF_MODEL_PATH.exists() and TFIDF_MATRIX_PATH.exists() and META_PATH.exists():
            with TFIDF_MODEL_PATH.open("rb") as fh:
                self.vectorizer = pickle.load(fh)
            with TFIDF_MATRIX_PATH.open("rb") as fh:
                self.tfidf_matrix = pickle.load(fh)
            with META_PATH.open("rb") as fh:
                self.metadata = pickle.load(fh)
            
            if not self.records and self.metadata:
                self.records = [AssessmentRecord.model_validate(item) for item in self.metadata]
                logger.warning(
                    "Catalog JSON empty; recovered %d records from metadata",
                    len(self.records),
                )
            return

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.records:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.metadata = []
            return

        docs = [
            " ".join(
                [
                    r.name,
                    r.description,
                    " ".join(r.skills),
                    " ".join(r.job_roles),
                    " ".join(r.tags),
                ]
            )
            for r in self.records
        ]
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        self.metadata = [r.model_dump(mode="json") for r in self.records]
        
        with TFIDF_MODEL_PATH.open("wb") as fh:
            pickle.dump(self.vectorizer, fh)
        with TFIDF_MATRIX_PATH.open("wb") as fh:
            pickle.dump(self.tfidf_matrix, fh)
        with META_PATH.open("wb") as fh:
            pickle.dump(self.metadata, fh)

    def search_vectors(self, query: str, top_k: int = 20) -> list[tuple[float, dict[str, Any]]]:
        if self.vectorizer is None or self.tfidf_matrix is None or not self.records:
            return []
        
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Sort in descending order
        top_indices = scores.argsort()[-min(top_k, len(self.records)):][::-1]
        
        out: list[tuple[float, dict[str, Any]]] = []
        for idx in top_indices:
            score = float(scores[idx])
            out.append((score, self.metadata[idx]))
        return out
