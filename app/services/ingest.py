from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Document:
    doc_id: int
    listing_id: int
    text: str
    district: str | None
    rooms: int | None
    area: float | None
    price_total_zl: float | None
    full_address: str | None
    url: str | None
    year_built: int | None


def _safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.,]", "", value).replace(",", ".")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_doc_text(row: pd.Series) -> str:
    parts: list[str] = []
    if pd.notna(row.get("city_district")):
        parts.append(f"Dzielnica: {row['city_district']}")
    if pd.notna(row.get("full_address")):
        parts.append(f"Adres: {row['full_address']}")
    if pd.notna(row.get("rooms")) or pd.notna(row.get("area")):
        rooms = row.get("rooms")
        area = row.get("area")
        parts.append(f"Pokoje: {rooms}, Metraż: {area} m2")
    price = _safe_float(row.get("price_total_zl"))
    if price is not None:
        parts.append(f"Cena: {int(price)} zł")
    year = _safe_int(row.get("year_built"))
    if year is not None:
        parts.append(f"Rok budowy: {year}")
    if pd.notna(row.get("description_text")):
        parts.append(f"Opis: {row['description_text']}")
    return "\n".join(parts)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start + 1, end)
            if split_at == -1:
                split_at = text.rfind(". ", start + 1, end)
            if split_at > start:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks or [text]


def load_documents(
    csv_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Brak pliku danych: {csv_path}")

    df = pd.read_csv(csv_path)
    documents: list[Document] = []
    doc_id = 0

    for listing_id, row in df.iterrows():
        base_text = build_doc_text(row)
        if not base_text.strip():
            continue

        chunks = _split_text(base_text, chunk_size, chunk_overlap)
        metadata = {
            "district": str(row["city_district"]) if pd.notna(row.get("city_district")) else None,
            "rooms": _safe_int(row.get("rooms")),
            "area": _safe_float(row.get("area")),
            "price_total_zl": _safe_float(row.get("price_total_zl")),
            "full_address": str(row["full_address"]) if pd.notna(row.get("full_address")) else None,
            "url": str(row["url"]) if pd.notna(row.get("url")) else None,
            "year_built": _safe_int(row.get("year_built")),
        }

        for chunk in chunks:
            documents.append(
                Document(
                    doc_id=doc_id,
                    listing_id=int(listing_id),
                    text=chunk,
                    district=metadata["district"],
                    rooms=metadata["rooms"],
                    area=metadata["area"],
                    price_total_zl=metadata["price_total_zl"],
                    full_address=metadata["full_address"],
                    url=metadata["url"],
                    year_built=metadata["year_built"],
                )
            )
            doc_id += 1

    return documents
