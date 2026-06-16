from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from app.schemas import SearchFilters
from app.services.embeddings import EmbeddingService
from app.services.ingest import Document, load_documents


@dataclass
class SearchResult:
    doc_id: int
    score: float
    text: str
    district: str | None
    rooms: int | None
    area: float | None
    price_total_zl: float | None
    full_address: str | None
    url: str | None


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: pd.DataFrame | None = None

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and self.metadata is not None

    @property
    def n_documents(self) -> int:
        if self.metadata is None:
            return 0
        return len(self.metadata)

    def index_paths(self, faiss_path: Path, meta_path: Path) -> tuple[Path, Path]:
        return faiss_path, meta_path

    def exists(self, faiss_path: Path, meta_path: Path) -> bool:
        return faiss_path.exists() and meta_path.exists()

    def build_from_csv(
        self,
        csv_path: Path,
        embedding_service: EmbeddingService,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        documents = load_documents(csv_path, chunk_size, chunk_overlap)
        if not documents:
            raise ValueError("Brak dokumentów do zaindeksowania")

        texts = [doc.text for doc in documents]
        vectors = embedding_service.embed_passages(texts)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        metadata = pd.DataFrame(
            [
                {
                    "doc_id": doc.doc_id,
                    "listing_id": doc.listing_id,
                    "text": doc.text,
                    "district": doc.district,
                    "rooms": doc.rooms,
                    "area": doc.area,
                    "price_total_zl": doc.price_total_zl,
                    "full_address": doc.full_address,
                    "url": doc.url,
                    "year_built": doc.year_built,
                }
                for doc in documents
            ]
        )

        self.index = index
        self.metadata = metadata

    def save(self, faiss_path: Path, meta_path: Path) -> None:
        if not self.is_loaded:
            raise RuntimeError("Indeks nie jest załadowany")
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        self.metadata.to_parquet(meta_path, index=False)

    def load(self, faiss_path: Path, meta_path: Path) -> None:
        if not self.exists(faiss_path, meta_path):
            raise FileNotFoundError("Brak zapisanych plików indeksu")
        self.index = faiss.read_index(str(faiss_path))
        self.metadata = pd.read_parquet(meta_path)

    def _apply_filters(self, filters: SearchFilters | None) -> pd.DataFrame:
        if self.metadata is None:
            raise RuntimeError("Metadane indeksu nie są załadowane")
        if filters is None:
            return self.metadata

        filtered = self.metadata.copy()
        if filters.district:
            needle = filters.district.casefold()
            filtered = filtered[
                filtered["district"].fillna("").str.casefold().str.contains(needle, regex=False)
            ]
        if filters.rooms is not None:
            filtered = filtered[filtered["rooms"] == filters.rooms]
        if filters.price_min is not None:
            filtered = filtered[
                filtered["price_total_zl"].notna() & (filtered["price_total_zl"] >= filters.price_min)
            ]
        if filters.price_max is not None:
            filtered = filtered[
                filtered["price_total_zl"].notna() & (filtered["price_total_zl"] <= filters.price_max)
            ]
        if filters.area_min is not None:
            filtered = filtered[filtered["area"].notna() & (filtered["area"] >= filters.area_min)]
        if filters.area_max is not None:
            filtered = filtered[filtered["area"].notna() & (filtered["area"] <= filters.area_max)]
        return filtered

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if not self.is_loaded:
            raise RuntimeError("Indeks nie jest załadowany")

        filtered = self._apply_filters(filters)
        if filtered.empty:
            return []

        allowed_ids = set(filtered["doc_id"].tolist())
        search_k = min(max(top_k * 5, top_k), self.n_documents)
        scores, indices = self.index.search(query_vector.reshape(1, -1), search_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            doc_id = int(idx)
            if doc_id not in allowed_ids:
                continue
            row = self.metadata.loc[self.metadata["doc_id"] == doc_id].iloc[0]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    text=str(row["text"]),
                    district=row["district"] if pd.notna(row["district"]) else None,
                    rooms=int(row["rooms"]) if pd.notna(row["rooms"]) else None,
                    area=float(row["area"]) if pd.notna(row["area"]) else None,
                    price_total_zl=float(row["price_total_zl"]) if pd.notna(row["price_total_zl"]) else None,
                    full_address=row["full_address"] if pd.notna(row["full_address"]) else None,
                    url=row["url"] if pd.notna(row["url"]) else None,
                )
            )
            if len(results) >= top_k:
                break
        return results
