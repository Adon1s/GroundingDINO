# tools/catalog_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    SentenceTransformer = None
    _ST_OK = False


_SEV_INT_TO_LABEL = {
    0: "none",
    1: "minor_repair",
    2: "moderate_repair",
    3: "full_replacement",
    4: "full_replacement",  # treat 4 as highest bucket for cost table compatibility
}


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _parse_severity_int(x: Any) -> int:
    # Accept int-like strings, otherwise map known labels
    if x is None:
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().lower()
    if s.isdigit():
        return int(s)
    label_to_int = {
        "none": 0,
        "minor_repair": 1,
        "moderate_repair": 2,
        "full_replacement": 3,
    }
    return label_to_int.get(s, 0)


def _catalog_text(item: Dict[str, Any]) -> str:
    name = item.get("name", item.get("id", ""))
    desc = item.get("description", "")
    cat = item.get("category", "")
    return _norm(f"{name}. {desc}. Category: {cat}.")


def _cosine_topk(q: np.ndarray, mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = mat @ q
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    k = max(1, min(int(k), n))
    if k == n:
        idxs = np.argsort(-scores)
    else:
        # kth should be k-1 (top-k selection)
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
    return scores[idxs], idxs


@dataclass
class MatchCandidate:
    id: str
    name: str
    category: str
    kind: str  # defect_issues | opportunity_flags
    score: float
    severity: int


class CatalogEmbedMatcher:
    """
    Builds embeddings for your catalog (defect_issues + opportunity_flags) and
    matches issues_natural_language descriptions to nearest catalog items.
    """

    def __init__(
        self,
        issue_catalog: Dict[str, Any],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        topk: int = 5,
        threshold_defect: float = 0.58,
        threshold_opportunity: float = 0.56,
        route_by_rough_category: bool = True,
        trust_remote_code: bool = False,
        device: str = "cpu",  # ✅ add this
    ):
        if not _ST_OK:
            raise RuntimeError("sentence-transformers not available. Install: pip install sentence-transformers")

        self.model_name = model_name
        self.topk = max(1, int(topk))
        self.threshold_defect = float(threshold_defect)
        self.threshold_opportunity = float(threshold_opportunity)
        self.route_by_rough_category = bool(route_by_rough_category)
        self.device = (device or "cpu").strip()  # ✅ store it

        # ✅ Create ST model and force device (compatible across ST versions)
        try:
            self._st = SentenceTransformer(
                self.model_name,
                trust_remote_code=trust_remote_code,
                device=self.device,  # (newer ST supports this)
            )
        except TypeError:
            # older ST: no `device=` kwarg
            self._st = SentenceTransformer(self.model_name, trust_remote_code=trust_remote_code)

        try:
            self._st.to(self.device)  # ✅ this is the actual “make it CPU” step
        except Exception as e:
            logger.warning(f"Could not move embeddings model to {self.device}: {e}")

        self._def_pack = self._build_pack(issue_catalog.get("defect_issues") or [], "defect_issues")
        self._opp_pack = self._build_pack(issue_catalog.get("opportunity_flags") or [], "opportunity_flags")
        self._all_pack = self._def_pack + self._opp_pack

        self._def_mat = self._embed_pack(self._def_pack)
        self._opp_mat = self._embed_pack(self._opp_pack)
        self._all_mat = self._embed_pack(self._all_pack)

    def _build_pack(self, items: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            out.append({
                "id": it["id"],
                "name": it.get("name", it["id"]),
                "category": it.get("category", "unknown"),
                "severity": _parse_severity_int(it.get("severity", 0)),
                "kind": kind,
                "text": _catalog_text(it),
            })
        return out

    def _embed_pack(self, pack: List[Dict[str, Any]]) -> np.ndarray:
        dim = int(getattr(self._st, "get_sentence_embedding_dimension", lambda: 384)())
        if not pack:
            return np.zeros((0, dim), dtype=np.float32)
        texts = [p["text"] for p in pack]
        vecs = self._st.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def match(self, issue_text: str, rough_category: Optional[str] = None) -> List[MatchCandidate]:
        t = _norm(issue_text)
        if not t:
            return []

        q = self._st.encode([t], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)

        # Routing: opportunities match opportunity catalog by default (optional)
        rough = (rough_category or "").strip().lower()
        if self.route_by_rough_category:
            if rough == "opportunity":
                pack, mat = self._opp_pack, self._opp_mat
            else:
                pack, mat = self._def_pack, self._def_mat
        else:
            pack, mat = self._all_pack, self._all_mat

        if mat.shape[0] == 0:
            return []

        scores, idxs = _cosine_topk(q, mat, self.topk)
        out: List[MatchCandidate] = []
        for s, i in zip(scores.tolist(), idxs.tolist()):
            meta = pack[i]
            out.append(MatchCandidate(
                id=meta["id"],
                name=meta["name"],
                category=meta["category"],
                kind=meta["kind"],
                score=float(s),
                severity=int(meta.get("severity", 0) or 0),
            ))
        return out

    def build_catalog_flags_and_annotate(
        self,
        issues_natural_language: List[Dict[str, Any]],
        attach_candidates: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns catalog_flags dict shaped exactly like your existing pipeline expects:
          { issue_id: {present, severity, evidence} }

        Also optionally annotates each issue dict with:
          - catalog_best_match
          - catalog_candidates
        """
        flags: Dict[str, Any] = {}

        for it in issues_natural_language or []:
            if not isinstance(it, dict):
                continue
            desc = _norm(str(it.get("description", "")))
            if not desc:
                continue

            rough = str(it.get("rough_category", "") or "")
            cands = self.match(desc, rough_category=rough)
            if not cands:
                if attach_candidates:
                    it["catalog_candidates"] = []
                    it.pop("catalog_best_match", None)
                continue

            best = cands[0]
            thr = self.threshold_opportunity if best.kind == "opportunity_flags" else self.threshold_defect
            present = "yes" if best.score >= thr else "uncertain"
            sev_label = _SEV_INT_TO_LABEL.get(best.severity, "none")
            if present != "yes":
                sev_label = "none"

            # Evidence: keep it simple and useful for debugging/UX
            loc = _norm(str(it.get("location_hint", "")))
            ev = desc if not loc else f"{desc} (location: {loc})"
            ev = f"{ev} | match={best.id} score={best.score:.3f}"

            # One flag per issue_id per photo - keep highest score if duplicate
            prev = flags.get(best.id)
            if prev is None:
                flags[best.id] = {"present": present, "severity": sev_label, "evidence": ev, "_score": best.score}
            else:
                # keep the better match for this photo
                if float(prev.get("_score", 0.0)) < best.score:
                    flags[best.id] = {"present": present, "severity": sev_label, "evidence": ev, "_score": best.score}

            if attach_candidates:
                it["catalog_candidates"] = [
                    {"id": c.id, "name": c.name, "category": c.category, "kind": c.kind, "score": c.score}
                    for c in cands
                ]
                if best.score >= thr:
                    it["catalog_best_match"] = {"id": best.id, "kind": best.kind, "score": best.score}
                else:
                    it.pop("catalog_best_match", None)

        # strip internal helper field
        for v in flags.values():
            if isinstance(v, dict):
                v.pop("_score", None)

        return flags