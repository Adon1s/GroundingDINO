# tools/catalog_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    SentenceTransformer = None
    _ST_OK = False


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _parse_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _cosine_topk(q: np.ndarray, mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    q: (d,), mat: (n,d) normalized embeddings.
    Returns (scores, idxs) for top-k cosine similarity (dot product).
    """
    if mat.shape[0] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    scores = mat @ q
    n = scores.shape[0]
    k = max(1, min(int(k), n))
    if k == n:
        idxs = np.argsort(-scores)
    else:
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
    return scores[idxs], idxs


@dataclass(frozen=True)
class CatalogItemMeta:
    item_id: str
    name: str
    kind: str              # defect | upgrade (safety/opportunity filtered out)
    trade_bucket: str
    severity: int
    text: str              # embed this


@dataclass(frozen=True)
class MatchCandidate:
    item_id: str
    name: str
    kind: str
    trade_bucket: str
    severity: int
    score: float


class CatalogEmbeddingsRetriever:
    """
    Embeddings-based candidate retrieval for catalog items.

    - Builds embeddings for catalog v2 items[].
    - Retrieves top-K candidates for each observation.
    - Does NOT decide present/absent (LLM resolver does that in Pass 2d).
    """

    def __init__(
        self,
        catalog_v2: Dict[str, Any],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        trust_remote_code: bool = False,
        default_topk: int = 10,
        # Optional: lexical guardrails keyed by item_id
        guardrails: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ):
        if not _ST_OK:
            raise RuntimeError("sentence-transformers not available. Install: pip install sentence-transformers")

        self.model_name = model_name
        self.device = (device or "cpu").strip()
        self.default_topk = max(1, int(default_topk))
        self.guardrails = guardrails or {}

        # Load model
        try:
            self._st = SentenceTransformer(
                self.model_name,
                trust_remote_code=trust_remote_code,
                device=self.device,
            )
        except TypeError:
            self._st = SentenceTransformer(self.model_name, trust_remote_code=trust_remote_code)

        try:
            self._st.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move embeddings model to {self.device}: {e}")

        # Build index
        items = catalog_v2.get("items") or []
        self._items: List[CatalogItemMeta] = self._build_items(items)
        self._mat: np.ndarray = self._embed_items(self._items)

        # Precompute indices by kind for fast filtering
        self._idx_by_kind: Dict[str, np.ndarray] = {}
        for i, meta in enumerate(self._items):
            self._idx_by_kind.setdefault(meta.kind, []).append(i)
        for k, idxs in self._idx_by_kind.items():
            self._idx_by_kind[k] = np.asarray(idxs, dtype=np.int64)

        # Convenience "slices" - only defect and upgrade (no safety/opportunity)
        self._defect_kinds = {"defect"}
        self._upgrade_kinds = {"upgrade"}

    def _catalog_text(self, it: Dict[str, Any]) -> str:
        # Embed the stable semantics: name + description + aliases + trade bucket
        name = str(it.get("name") or it.get("id") or it.get("defect_id") or "").strip()
        desc = str(it.get("description") or "").strip()
        aliases = it.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        trade = str(it.get("trade_bucket") or "").strip()
        kind = str(it.get("kind") or "").strip()

        blob = f"{name}. {desc}."
        if aliases:
            blob += " Aliases: " + "; ".join(aliases) + "."
        if trade:
            blob += f" Trade: {trade}."
        if kind:
            blob += f" Kind: {kind}."
        return _norm(blob)

    def _build_items(self, items: List[Dict[str, Any]]) -> List[CatalogItemMeta]:
        out: List[CatalogItemMeta] = []
        for it in items:
            if not isinstance(it, dict):
                continue

            # Accept id, defect_id, or upgrade_id as the identifier
            item_id = str(it.get("id") or it.get("defect_id") or it.get("upgrade_id") or "").strip()
            if not item_id:
                continue

            name = str(it.get("name") or item_id).strip()
            kind = str(it.get("kind") or "defect").strip().lower()
            trade_bucket = str(it.get("trade_bucket") or "").strip().lower()
            severity = _parse_int(it.get("severity"), default=0)

            # Filter out safety and opportunity kinds - they should not appear downstream
            if kind in {"safety", "opportunity"}:
                continue

            text = self._catalog_text(it)

            out.append(CatalogItemMeta(
                item_id=item_id,
                name=name,
                kind=kind,
                trade_bucket=trade_bucket,
                severity=severity,
                text=text,
            ))
        return out

    def _embed_items(self, items: List[CatalogItemMeta]) -> np.ndarray:
        dim = int(getattr(self._st, "get_sentence_embedding_dimension", lambda: 384)())
        if not items:
            return np.zeros((0, dim), dtype=np.float32)
        texts = [m.text for m in items]
        vecs = self._st.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def _passes_guardrails(self, text: str, item_id: str) -> bool:
        g = self.guardrails.get(item_id)
        if not g:
            return True
        t = text.lower()
        deny = g.get("deny_any", [])
        if deny and any(x in t for x in deny):
            return False
        must = g.get("must_any", [])
        if must and not any(x in t for x in must):
            return False
        return True

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        texts = [_norm(t) for t in texts]
        vecs = self._st.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def retrieve_candidates(
        self,
        observation_text: str,
        *,
        topk: Optional[int] = None,
        allowed_kinds: Optional[Set[str]] = None,
    ) -> List[MatchCandidate]:
        t = _norm(observation_text)
        if not t or self._mat.shape[0] == 0:
            return []

        q = self._encode_queries([t])[0]

        idxs = None
        if allowed_kinds:
            # build a union of indices for those kinds
            buf: List[int] = []
            for k in allowed_kinds:
                arr = self._idx_by_kind.get(k)
                if arr is not None and arr.size:
                    buf.extend(arr.tolist())
            idxs = np.asarray(sorted(set(buf)), dtype=np.int64) if buf else None

        mat = self._mat if idxs is None else self._mat[idxs, :]
        pack = self._items if idxs is None else [self._items[i] for i in idxs.tolist()]

        k = topk if topk is not None else self.default_topk
        scores, rel = _cosine_topk(q, mat, k)

        out: List[MatchCandidate] = []
        for s, i in zip(scores.tolist(), rel.tolist()):
            meta = pack[i]
            if not self._passes_guardrails(t, meta.item_id):
                continue
            out.append(MatchCandidate(
                item_id=meta.item_id,
                name=meta.name,
                kind=meta.kind,
                trade_bucket=meta.trade_bucket,
                severity=meta.severity,
                score=float(s),
            ))
        return out

    # Convenience wrappers that make your pipeline code very explicit
    def embeddings_retrieve_defect_candidates(self, observation_text: str, topk: Optional[int] = None) -> List[MatchCandidate]:
        return self.retrieve_candidates(observation_text, topk=topk, allowed_kinds=self._defect_kinds)

    def embeddings_retrieve_upgrade_candidates(self, observation_text: str, topk: Optional[int] = None) -> List[MatchCandidate]:
        return self.retrieve_candidates(observation_text, topk=topk, allowed_kinds=self._upgrade_kinds)