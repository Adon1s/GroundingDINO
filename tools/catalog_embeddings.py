# tools/catalog_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import logging
import numpy as np

from tools.pipeline_common import SCENE_GROUPS_UI, term_matches

logger = logging.getLogger(__name__)

# All known scene groups — used as the fallback when a catalog item has no scene_groups field.
_ALL_SCENE_GROUPS = tuple(SCENE_GROUPS_UI)

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    SentenceTransformer = None
    _ST_OK = False


# =============================================================================
# Embeddings backends
# -----------------------------------------------------------------------------
# The retriever encodes catalog items and observations through a pluggable
# encoder. Two backends are provided:
#   - OpenAICompatibleEncoder  (default) — a local llama-server sidecar serving
#     the Q8_0 GGUF text-matching model over /v1/embeddings.
#   - SentenceTransformerEncoder — the legacy in-process path, kept for tests
#     and one-flag rollback.
# Every encoder returns an (n, dimension) L2-normalized float32 ndarray so the
# existing dot-product cosine retrieval is unchanged.
# =============================================================================

# Local llama-server needs no auth, but the OpenAI SDK still requires a key.
_LOCAL_API_KEY_PLACEHOLDER = "local-no-auth-placeholder"
_HTTP_BATCH_SIZE = 32
_HTTP_TIMEOUT_SECONDS = 60.0


class EmbeddingsRuntimeError(RuntimeError):
    """Raised when an embeddings backend is unavailable or returns malformed output."""


class SentenceTransformerEncoder:
    """In-process SentenceTransformer encoder (legacy backend + tests/rollback)."""

    def __init__(self, model_name: str, device: str = "cpu", trust_remote_code: bool = False):
        if not _ST_OK:
            raise EmbeddingsRuntimeError(
                "sentence-transformers not available. Install: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = (device or "cpu").strip()
        try:
            self._st = SentenceTransformer(
                model_name, trust_remote_code=trust_remote_code, device=self.device
            )
        except TypeError:
            self._st = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        try:
            self._st.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move embeddings model to {self.device}: {e}")
        self.dimension = int(getattr(self._st, "get_sentence_embedding_dimension", lambda: 384)())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._st.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatibleEncoder:
    """Encoder backed by a local OpenAI-compatible /v1/embeddings server (llama-server).

    Catalog and observation text are encoded identically (no query/document prefix).
    Response order is restored via each result's ``index``; outputs are converted to
    float32, required to be exactly ``dimension`` finite non-zero values, and
    L2-normalized before being handed to the existing dot-product scoring.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dimension: int = 1024,
        api_key: str = _LOCAL_API_KEY_PLACEHOLDER,
        batch_size: int = _HTTP_BATCH_SIZE,
        timeout: float = _HTTP_TIMEOUT_SECONDS,
    ):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbeddingsRuntimeError(
                f"openai package not available for the openai_compatible backend: {exc}"
            ) from exc
        self.model_name = model_name
        self.dimension = int(dimension)
        self.batch_size = max(1, int(batch_size))
        # Local server does not require auth; api_key is a placeholder the SDK expects.
        # max_retries=0: a local sidecar is either up or not — fail fast and clearly
        # instead of the SDK silently retrying a refused connection for ~9s.
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out = np.empty((len(texts), self.dimension), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            try:
                resp = self._client.embeddings.create(model=self.model_name, input=batch)
            except Exception as exc:
                # Connection errors, timeouts, HTTP errors, etc.
                raise EmbeddingsRuntimeError(
                    f"embeddings request failed (model={self.model_name}): {exc}"
                ) from exc

            data = getattr(resp, "data", None)
            if data is None or len(data) != len(batch):
                got = 0 if data is None else len(data)
                raise EmbeddingsRuntimeError(
                    f"embeddings response returned {got} vectors for {len(batch)} inputs"
                )

            seen: Set[int] = set()
            for item in data:
                idx = getattr(item, "index", None)
                if not isinstance(idx, int) or idx < 0 or idx >= len(batch):
                    raise EmbeddingsRuntimeError(
                        f"embeddings response has invalid index {idx!r} for batch of {len(batch)}"
                    )
                if idx in seen:
                    raise EmbeddingsRuntimeError(f"embeddings response has duplicate index {idx}")
                seen.add(idx)

                vec = np.asarray(getattr(item, "embedding", []), dtype=np.float32)
                if vec.shape != (self.dimension,):
                    raise EmbeddingsRuntimeError(
                        f"embeddings vector has shape {tuple(vec.shape)}, "
                        f"expected ({self.dimension},)"
                    )
                if not np.all(np.isfinite(vec)):
                    raise EmbeddingsRuntimeError("embeddings vector contains NaN or Infinity")
                norm = float(np.linalg.norm(vec))
                if norm == 0.0 or not np.isfinite(norm):
                    raise EmbeddingsRuntimeError(
                        "embeddings vector is zero or has a non-finite norm"
                    )
                out[start + idx] = vec / norm

        return out


def _build_encoder(
    *,
    backend: str,
    model_name: str,
    base_url: Optional[str],
    device: str,
    trust_remote_code: bool,
    st_model_name: Optional[str],
    embedding_dimension: int,
):
    """Construct the encoder for the requested backend. Never silently falls back."""
    b = (backend or "").strip().lower()
    if b in ("openai_compatible", "openai", "http", "llama_server", "llama-server"):
        if not base_url:
            raise EmbeddingsRuntimeError(
                "base_url (EMBEDDINGS_BASE_URL) is required for the openai_compatible backend"
            )
        return OpenAICompatibleEncoder(
            base_url=base_url,
            model_name=model_name,
            dimension=embedding_dimension,
        )
    if b in ("sentence_transformer", "sentence-transformers", "st", "sbert"):
        return SentenceTransformerEncoder(
            model_name=st_model_name or model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    raise EmbeddingsRuntimeError(f"Unknown embeddings backend: {backend!r}")


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


def build_guardrails_from_catalog(catalog: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Build hard/soft keyword constraints from catalog.

    Catalog keyword schema (three-tier):
      deny_any:    if any term matches, reject the candidate
      require_any: at least one term must match, or reject
      support_any: soft support / ranking hint only (not enforced as gate)
    """
    guardrails: Dict[str, Dict[str, List[str]]] = {}
    for item in catalog.get("items", []):
        item_id = str(item.get("id") or item.get("defect_id") or item.get("upgrade_id") or "").strip()
        if not item_id:
            continue
        deny = [t.lower() for t in item.get("deny_any", []) if str(t).strip()]
        support = [t.lower() for t in item.get("support_any", []) if str(t).strip()]
        require = [t.lower() for t in item.get("require_any", []) if str(t).strip()]
        if deny or support or require:
            guardrails[item_id] = {
                "deny_any": deny,
                "support_any": support,
                "require_any": require,
            }
    return guardrails


@dataclass(frozen=True)
class CatalogItemMeta:
    item_id: str
    name: str
    kind: str              # as authored in the catalog (validator owns the enum)
    trade_bucket: str
    severity: int
    description: str
    support_any: Tuple[str, ...]
    default_hidden: bool
    drop_if_generic: bool
    scene_groups: Tuple[str, ...]  # scene groups this item is valid for (from catalog)
    text: str              # embed this


@dataclass(frozen=True)
class MatchCandidate:
    item_id: str
    name: str
    kind: str
    trade_bucket: str
    severity: int
    description: str
    support_any: Tuple[str, ...]
    defaultHidden: bool
    drop_if_generic: bool
    score: float
    scene_groups: Tuple[str, ...]  # passed through from CatalogItemMeta


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
        *,
        backend: str = "openai_compatible",
        base_url: Optional[str] = None,
        st_model_name: Optional[str] = None,
        embedding_dimension: int = 1024,
        encoder: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.device = (device or "cpu").strip()
        self.default_topk = max(1, int(default_topk))
        self.guardrails = guardrails or {}

        # Build the encoder backend. Tests may inject a deterministic fake via `encoder`.
        if encoder is not None:
            self._encoder = encoder
        else:
            self._encoder = _build_encoder(
                backend=backend,
                model_name=model_name,
                base_url=base_url,
                device=self.device,
                trust_remote_code=trust_remote_code,
                st_model_name=st_model_name,
                embedding_dimension=embedding_dimension,
            )

        # Build index. For the HTTP backend, embedding the catalog here doubles as the
        # server-availability + output-shape check (raises EmbeddingsRuntimeError on failure).
        items = catalog_v2.get("items") or []
        self._items: List[CatalogItemMeta] = self._build_items(items)
        self._mat: np.ndarray = self._embed_items(self._items)

        # Precompute indices by kind for fast filtering
        self._idx_by_kind: Dict[str, np.ndarray] = {}
        for i, meta in enumerate(self._items):
            self._idx_by_kind.setdefault(meta.kind, []).append(i)
        for k, idxs in self._idx_by_kind.items():
            self._idx_by_kind[k] = np.asarray(idxs, dtype=np.int64)

        # Precompute indices by scene group.
        # Built from each item's meta.scene_groups (read from catalog JSON) so that
        # per-item overrides in the catalog are the single source of truth.
        # Items without a scene_groups entry default to _ALL_SCENE_GROUPS in _build_items,
        # so they remain reachable under any group filter.
        self._idx_by_group: Dict[str, np.ndarray] = {}
        tmp: Dict[str, List[int]] = {}
        for i, meta in enumerate(self._items):
            for g in meta.scene_groups:
                tmp.setdefault(g, []).append(i)
        for g, idxs in tmp.items():
            self._idx_by_group[g] = np.asarray(sorted(set(idxs)), dtype=np.int64)


    def _catalog_text(self, it: Dict[str, Any]) -> str:
        # Replace semantics: a non-empty string `embed_text` is the full embedding
        # source. The catalog auditor treats embed_text as the canonical place to
        # tune retrieval, so when it's set we use it verbatim (after normalization).
        raw_embed = it.get("embed_text")
        if isinstance(raw_embed, str) and raw_embed.strip():
            return _norm(raw_embed)

        # Fallback: original composition (name + description + aliases + trade + kind).
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
            kind = str(it.get("kind") or "").strip().lower()
            if not kind:
                # No silent kind coercion: an item without a kind is a catalog
                # authoring error (validator-enforced), not a defect by default.
                logger.warning("catalog item %r has no kind; skipping from index", item_id)
                continue
            trade_bucket = str(it.get("trade_bucket") or "").strip().lower()
            severity = _parse_int(it.get("severity"), default=0)
            description = str(it.get("description") or "").strip()
            support_any = tuple(str(x).strip() for x in (it.get("support_any") or []) if str(x).strip())
            default_hidden = bool(it.get("defaultHidden", False))
            drop_if_generic = bool(it.get("drop_if_generic", False))

            # scene_groups: read directly from catalog; fall back to all groups if missing
            raw_sg = it.get("scene_groups")
            if isinstance(raw_sg, list) and raw_sg:
                scene_groups: Tuple[str, ...] = tuple(str(s).strip() for s in raw_sg if str(s).strip())
            else:
                scene_groups = tuple(_ALL_SCENE_GROUPS)

            text = self._catalog_text(it)

            out.append(CatalogItemMeta(
                item_id=item_id,
                name=name,
                kind=kind,
                trade_bucket=trade_bucket,
                severity=severity,
                description=description,
                support_any=support_any,
                default_hidden=default_hidden,
                drop_if_generic=drop_if_generic,
                scene_groups=scene_groups,
                text=text,
            ))
        return out

    def _embed_items(self, items: List[CatalogItemMeta]) -> np.ndarray:
        if not items:
            return np.zeros((0, int(self._encoder.dimension)), dtype=np.float32)
        texts = [m.text for m in items]
        return self._encoder.encode(texts)

    def _passes_guardrails(self, text: str, item_id: str) -> bool:
        g = self.guardrails.get(item_id)
        if not g:
            return True
        t = text.lower()
        deny = g.get("deny_any", [])
        if deny and any(term_matches(x, t) for x in deny):
            return False
        require = g.get("require_any", [])
        if require and not any(term_matches(x, t) for x in require):
            return False
        return True

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        normed = [_norm(t) for t in texts]
        return self._encoder.encode(normed)

    def retrieve_candidates(
        self,
        observation_text: str,
        *,
        topk: Optional[int] = None,
        allowed_kinds: Optional[Set[str]] = None,
        allowed_groups: Optional[Set[str]] = None,
    ) -> List[MatchCandidate]:
        """
        Retrieve top-K candidates for an observation.

        Args:
            observation_text: The issue description to match.
            topk:             Override default_topk.
            allowed_kinds:    None means deliberately unfiltered. Any other value
                              restricts to items whose kind is in the set — an
                              empty set, or a set of kinds absent from the index,
                              returns NO candidates (never the whole catalog).
            allowed_groups:   If set, restrict to items whose scene_groups (from catalog)
                              overlaps this set. Pass the photo's scene group here to
                              prevent cross-room matches (e.g. kitchen items in bathroom).
        """
        t = _norm(observation_text)
        if not t or self._mat.shape[0] == 0:
            return []

        q = self._encode_queries([t])[0]

        idxs: Optional[np.ndarray] = None

        # Kind filter — union of per-kind index arrays. `is not None` matters:
        # an empty set must fail closed to [], not fall through as "no filter".
        if allowed_kinds is not None:
            buf: List[int] = []
            for k in allowed_kinds:
                arr = self._idx_by_kind.get(k)
                if arr is not None and arr.size:
                    buf.extend(arr.tolist())
            idxs = np.asarray(sorted(set(buf)), dtype=np.int64) if buf else np.array([], dtype=np.int64)
            if idxs.size == 0:
                return []

        # Group filter — union of per-group arrays, then intersect with kind filter
        if allowed_groups:
            gbuf: List[int] = []
            for g in allowed_groups:
                arr = self._idx_by_group.get(g)
                if arr is not None and arr.size:
                    gbuf.extend(arr.tolist())
            gidxs = np.asarray(sorted(set(gbuf)), dtype=np.int64) if gbuf else None

            if gidxs is None:
                # No items indexed for these groups → nothing to return
                return []
            if idxs is None:
                idxs = gidxs
            else:
                idxs = np.intersect1d(idxs, gidxs, assume_unique=False)
                if idxs.size == 0:
                    return []

        mat  = self._mat  if idxs is None else self._mat[idxs, :]
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
                description=meta.description,
                support_any=meta.support_any,
                defaultHidden=meta.default_hidden,
                drop_if_generic=meta.drop_if_generic,
                score=float(s),
                scene_groups=meta.scene_groups,
            ))
        return out


def make_candidate_provider(retriever: "CatalogEmbeddingsRetriever") -> Any:
    """
    Wrap a retriever as the Pass 2d candidate-provider callable.

    Strict exact-kind semantics (observation-kind-v2): kind membership is
    decided by the retriever's index, never by a hardcoded vocabulary, so the
    same closure serves any catalog. Filter contract:

    - context carries an explicit ``allowed_kinds`` (even an empty one) — use
      it verbatim; empty/unknown kinds fail closed to zero candidates.
    - otherwise a single ``kind`` — search exactly that kind.
    - neither — deliberately unfiltered (None).
    """
    from dataclasses import asdict

    def candidate_provider(observation_text: str, context: dict) -> list:
        kind = (context.get("kind") or "").strip().lower()
        topk = context.get("top_k_candidates")
        scene_group = context.get("scene_group")
        allowed_groups = {scene_group} if scene_group else None
        if "allowed_kinds" in context:
            allowed_kinds: Optional[Set[str]] = {
                str(k).strip().lower() for k in (context.get("allowed_kinds") or [])
            }
        elif kind:
            allowed_kinds = {kind}
        else:
            allowed_kinds = None
        matches = retriever.retrieve_candidates(
            observation_text,
            topk=topk,
            allowed_kinds=allowed_kinds,
            allowed_groups=allowed_groups,
        )
        return [asdict(m) for m in matches]

    return candidate_provider


def build_candidate_provider(catalog: Dict[str, Any]) -> Any:
    """
    Build the Pass 2d candidate provider backed by the embeddings sidecar.

    Deliberately has no exception handling. The encoder is fail-fast by design
    (max_retries=0, catalog embedding at construction doubles as an availability
    and output-shape probe), and callers must let EmbeddingsRuntimeError reach
    their preflight. Swallowing it yields a run that resolves zero catalog items
    and reports success -- the exact failure this contract exists to prevent.

    Callers that intend to run without Pass 2d must skip this entirely rather
    than tolerate a failure from it.
    """
    from tools import pipeline_config as cfg

    retriever = CatalogEmbeddingsRetriever(
        catalog_v2=catalog,
        model_name=getattr(cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        device=getattr(cfg, "EMBEDDINGS_DEVICE", "cpu"),
        trust_remote_code=getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False),
        default_topk=getattr(cfg, "EMBEDDINGS_TOPK", 10),
        guardrails=build_guardrails_from_catalog(catalog),
        backend=getattr(cfg, "EMBEDDINGS_BACKEND", "openai_compatible"),
        base_url=getattr(cfg, "EMBEDDINGS_BASE_URL", "http://127.0.0.1:8081/v1"),
        st_model_name=getattr(cfg, "EMBEDDINGS_ST_MODEL_NAME", "jinaai/jina-embeddings-v3"),
        embedding_dimension=getattr(cfg, "EMBEDDINGS_DIMENSION", 1024),
    )

    logger.info(
        "Pass 2d candidate_provider ready (model=%s, items=%d)",
        retriever.model_name,
        len(retriever._items),
    )
    return make_candidate_provider(retriever)
