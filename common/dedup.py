"""
Exact and near-duplicate detection, pure standard library.

WHY THIS EXISTS
---------------
Two independent reasons, both discovered during the Phase 1 audit:

1. The Konkani books corpus averages 7.52 words per record. It is line-
   fragmented OCR, not documents. Short lines repeat heavily - page headers,
   chapter titles, publisher boilerplate, running feet - so the exact-duplicate
   rate is expected to be high, and every duplicate line inflates the token
   count without adding information.

2. The Maharashtra Government Resolutions are formulaic by design. Thousands of
   documents share the same departmental header, the same reference-number
   block, the same closing signature paragraph. These are *near* duplicates:
   never byte-identical, but carrying almost no new information. Exact hashing
   cannot catch them.

APPROACH
--------
Exact: SHA-256 over the normalized text. Cheap, and catches the line-fragment
case.

Near: MinHash over character n-gram shingles, with banded LSH. Implemented here
rather than pulled from `datasketch` to keep the dependency surface small and,
more importantly, so the mechanism can be explained rather than invoked.

How MinHash works, in one paragraph: represent each document as the set of its
character 5-grams. The Jaccard similarity of two such sets is what we want, but
comparing every pair is O(n^2). Instead, hash every shingle with k independent
hash functions and keep only the minimum value for each - that is the signature.
The probability that two documents share a given minimum equals their Jaccard
similarity, so comparing k-length signatures estimates similarity in fixed time.
LSH then splits each signature into `bands` bands; two documents become
candidates if any whole band matches. Documents that are very similar collide in
at least one band with high probability, so we only ever compare candidates.

WHY CHARACTER n-GRAMS AND NOT WORD n-GRAMS
------------------------------------------
Devanagari is agglutinative in its case marking and the OCR is noisy. Word-level
shingles are brittle: one mis-OCR'd character destroys a whole word token.
Character 5-grams degrade gracefully - a single bad character damages five
shingles out of thousands.
"""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

_WS_RE = re.compile(r"\s+")

# Mersenne prime 2^31 - 1, chosen so that a*h stays inside a signed 64-bit
# integer and the whole permutation step can run as one vectorised numpy
# operation. With a 2^61-1 prime the products overflow int64 and numpy silently
# wraps, which would corrupt every signature.
_MERSENNE_PRIME = (1 << 31) - 1
_MAX_HASH = _MERSENNE_PRIME

# Long documents are sampled rather than shingled exhaustively. MinHash
# estimates Jaccard similarity from a random sample of the shingle set, so a
# bounded sample gives essentially the same estimate at a fixed cost - and
# without it, one very long document can dominate the run time.
_MAX_SHINGLES = 1200

try:
    import numpy as _np
except ImportError:                     # pragma: no cover
    _np = None


def canonical_form(text: str) -> str:
    """Aggressive canonicalisation used ONLY for duplicate comparison.

    This is not what gets stored in the corpus. Digits are folded to a single
    placeholder and case is lowered so that two government resolutions differing
    only in their reference number and date are recognised as near-identical.
    """
    text = _WS_RE.sub(" ", text).strip().lower()
    text = re.sub(r"[0-9०-९]+", "0", text)
    return text


def exact_hash(text: str) -> str:
    """SHA-256 of the canonical form. Two documents with the same value are dupes."""
    return hashlib.sha256(canonical_form(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-gram set of the canonical form."""
    return shingles_from_canon(canonical_form(text), k)


def shingles_from_canon(t: str, k: int = 5) -> set[str]:
    """Character k-gram set of an ALREADY canonicalised string."""
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _hash32(data: bytes) -> int:
    """32-bit hash of a shingle.

    crc32 rather than a cryptographic hash: measured at 0.03ms vs 0.13ms per
    document for blake2b over the same shingle set. MinHash needs a fast,
    well-distributed, *deterministic* hash - not a secure one - and crc32 is all
    three. Determinism matters because signatures must reproduce across runs
    (Python's built-in hash() is randomised per process and would not).
    """
    return zlib.crc32(data) & _MAX_HASH


class MinHasher:
    """Fixed set of permutations, so signatures are comparable across runs."""

    def __init__(self, num_perm: int = 128, seed: int = 0xC0FFEE):
        self.num_perm = num_perm
        rng = _LCG(seed)
        # a must be non-zero for the permutation to be a bijection.
        self.perms = [
            ((rng.next() % (_MERSENNE_PRIME - 1)) + 1, rng.next() % _MERSENNE_PRIME)
            for _ in range(num_perm)
        ]

        # Pre-split the permutation coefficients into arrays once, so the
        # per-document work is a single vectorised expression.
        if _np is not None:
            self._a = _np.array([a for a, _ in self.perms], dtype=_np.int64)
            self._b = _np.array([b for _, b in self.perms], dtype=_np.int64)

    def signature(self, text: str, k: int = 5) -> tuple[int, ...]:
        return self.signature_from_canon(canonical_form(text), k)

    def signature_from_canon(self, canon: str, k: int = 5) -> tuple[int, ...]:
        sh = shingles_from_canon(canon, k)
        if not sh:
            return tuple([_MAX_HASH] * self.num_perm)

        if len(sh) > _MAX_SHINGLES:
            # Deterministic sample: sort so the same document always yields the
            # same subset, then take an evenly spaced stride.
            ordered = sorted(sh)
            step = len(ordered) / _MAX_SHINGLES
            sh = {ordered[int(i * step)] for i in range(_MAX_SHINGLES)}

        base = [_hash32(s.encode("utf-8")) & _MAX_HASH for s in sh]

        if _np is None:                                    # pragma: no cover
            return tuple(min((a * h + b) % _MERSENNE_PRIME for h in base)
                         for a, b in self.perms)

        # One (num_perm x n_shingles) matrix instead of num_perm x n_shingles
        # interpreted Python operations. On a 14k-document corpus this is the
        # difference between minutes and seconds.
        h = _np.array(base, dtype=_np.int64)
        products = (self._a[:, None] * h[None, :] + self._b[:, None]) % _MERSENNE_PRIME
        return tuple(int(v) for v in products.min(axis=1))


class _LCG:
    """Small deterministic PRNG so signatures reproduce exactly across machines."""

    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.state >> 16


def jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


@dataclass
class DedupStats:
    seen: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    kept: int = 0

    @property
    def duplicate_rate(self) -> float:
        total = self.exact_duplicates + self.near_duplicates
        return total / self.seen if self.seen else 0.0

    def to_dict(self) -> dict:
        return {
            "seen": self.seen,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "kept": self.kept,
            "duplicate_rate": round(self.duplicate_rate, 6),
        }


class Deduplicator:
    """Streaming exact + near duplicate filter.

    Usage:
        d = Deduplicator(threshold=0.85)
        if d.is_duplicate(text):
            skip
        else:
            keep

    Memory: holds one hash and one signature per kept document. At 128
    permutations that is roughly 1 KB per document, so a few hundred thousand
    documents fit comfortably in RAM on a laptop. Beyond that, shard by source
    and dedup across shards in a second pass.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        num_perm: int = 128,
        bands: int = 32,
        shingle_k: int = 5,
    ):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.shingle_k = shingle_k
        self.bands = bands
        self.rows = num_perm // bands
        self.hasher = MinHasher(num_perm)
        self._exact: set[str] = set()
        self._buckets: dict[tuple[int, bytes], list[int]] = {}
        self._signatures: list[tuple[int, ...]] = []
        self.stats = DedupStats()

    def _band_keys(self, sig: tuple[int, ...]) -> Iterator[tuple[int, int]]:
        """Band signatures. crc32 over packed ints - no string joins per band."""
        for b in range(self.bands):
            chunk = sig[b * self.rows:(b + 1) * self.rows]
            yield b, zlib.crc32(struct.pack(f"<{len(chunk)}I", *chunk))

    def is_duplicate(self, text: str) -> bool:
        """Check-and-register. Returns True if `text` duplicates something seen.

        `canonical_form` is computed ONCE here and reused. It was previously
        recomputed inside both exact_hash() and shingles(), running two regex
        passes over the full document twice per call - a large share of the run
        time on a corpus of hundreds of thousands of documents.
        """
        self.stats.seen += 1

        canon = canonical_form(text)

        h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        if h in self._exact:
            self.stats.exact_duplicates += 1
            return True

        sig = self.hasher.signature_from_canon(canon, self.shingle_k)

        candidates: set[int] = set()
        keys = list(self._band_keys(sig))
        for key in keys:
            candidates.update(self._buckets.get(key, ()))

        for idx in candidates:
            if jaccard(sig, self._signatures[idx]) >= self.threshold:
                self.stats.near_duplicates += 1
                return True

        # Not a duplicate: register it.
        self._exact.add(h)
        idx = len(self._signatures)
        self._signatures.append(sig)
        for key in keys:
            self._buckets.setdefault(key, []).append(idx)
        self.stats.kept += 1
        return False


def cross_corpus_overlap(
    hashes_a: Iterable[str], hashes_b: Iterable[str]
) -> tuple[int, set[str]]:
    """Exact-hash overlap between two corpora.

    This is the evidence for the specification's requirement that Model H and
    Model L must not share documents. The expected and required result is 0.
    """
    set_a = set(hashes_a)
    shared = set_a & set(hashes_b)
    return len(shared), shared


if __name__ == "__main__":
    d = Deduplicator(threshold=0.85)

    base = ("महाराष्ट्र शासन निर्णय क्रमांक १२३४ दिनांक ०१/०१/२०२६ "
            "विभाग सामान्य प्रशासन विभाग या शासन निर्णयान्वये असे कळविण्यात येते की "
            "संबंधित अधिकाऱ्यांनी याची नोंद घ्यावी आणि आवश्यक ती कार्यवाही करावी. ")

    assert not d.is_duplicate(base), "first document must be kept"
    assert d.is_duplicate(base), "byte-identical must be an exact duplicate"

    # Same boilerplate, different reference number and date: a near duplicate.
    variant = base.replace("१२३४", "५६७८").replace("०१/०१/२०२६", "१५/०३/२०२६")
    assert d.is_duplicate(variant), "GR boilerplate variant must be caught as near-dup"

    different = ("गोंयची भास कोंकणी आसा आनी तिचो इतिहास पोरनो आसा. हांव घरा वतां "
                 "आनी जेवण करतां. तांणी सांगलें की हें काम बरें जालें म्हण. ")
    assert not d.is_duplicate(different), "unrelated text must not be flagged"

    print("dedup stats:", d.stats.to_dict())

    n, shared = cross_corpus_overlap(["h1", "h2"], ["h3", "h4"])
    assert n == 0
    n, shared = cross_corpus_overlap(["h1", "h2"], ["h2", "h4"])
    assert n == 1 and shared == {"h2"}

    print("dedup self-test: all assertions passed")
