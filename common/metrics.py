#!/usr/bin/env python3
"""
Generation-quality metrics, implemented directly rather than imported.

WHY NOT sacrebleu / nltk / rouge-score
--------------------------------------
Two reasons, one practical and one that matters more.

Practical: the Kaggle notebook runs with internet disabled, so anything not in
the base image cannot be installed. Depending on a package that may or may not be
present is a way to discover a missing dependency an hour into an evaluation run.

The one that matters: these three metrics are the ones the specification asks us
to interpret - "briefly explain why each metric is or is not informative for
open-ended LM generation in your languages". Writing them out means knowing
exactly what they count, which is what makes that explanation possible rather
than recited.

WHAT EACH METRIC ACTUALLY MEASURES
----------------------------------
BLEU-4: modified n-gram precision for n = 1..4, geometric mean, times a brevity
penalty. It asks "what fraction of the model's n-grams appear in the reference",
clipped so repeating a correct word ten times does not score ten times. It is
precision-oriented and word-level, which makes it a poor fit for open-ended
generation in a morphologically rich language: a fluent Devanagari continuation
that inflects a word differently from the reference scores zero on that token,
even though a reader would accept both.

chrF: the same idea over *character* n-grams (n = 1..6), combining precision and
recall as an F-score with beta = 2, so recall is weighted twice as heavily. This
is the least bad of the three here. Marathi and Konkani are agglutinative enough
that word-level matching is brittle, and character n-grams give partial credit
for a correct stem with a different suffix.

ROUGE-L: F-measure over the longest common subsequence. Unlike BLEU it does not
require contiguity, so it rewards getting the *order* of content right even when
words are inserted between. It is recall-oriented, which is why it is usually
reported alongside BLEU rather than instead of it.

All three compare against a single reference continuation. For open-ended
generation there are many acceptable continuations, so absolute values will be
low for every model and only the *comparison* between Model H and Model L carries
information.
"""

from __future__ import annotations

import math
from collections import Counter


def ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token sequence."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------- BLEU

def corpus_bleu(hypotheses: list[list[str]], references: list[list[str]],
                max_n: int = 4) -> dict:
    """Corpus-level BLEU-4 with the standard brevity penalty.

    Corpus-level, not the mean of sentence scores: n-gram counts are pooled
    across the whole corpus before the precision is taken. Sentence-level BLEU
    needs smoothing to avoid zeros on short outputs, and averaging smoothed
    sentence scores is a different quantity from the corpus score.
    """
    clipped = [0] * max_n          # matched n-grams, clipped to reference counts
    total = [0] * max_n            # candidate n-grams
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h_counts = ngrams(hyp, n)
            r_counts = ngrams(ref, n)
            # Clipping: a candidate n-gram can be credited at most as many times
            # as it appears in the reference. Without it, emitting one correct
            # word 50 times would score perfectly.
            clipped[n - 1] += sum(min(c, r_counts[g]) for g, c in h_counts.items())
            total[n - 1] += max(0, len(hyp) - n + 1)

    precisions = [(clipped[i] / total[i]) if total[i] else 0.0
                  for i in range(max_n)]

    # A zero at any order makes the geometric mean zero. That is the defined
    # behaviour of corpus BLEU and is reported rather than smoothed away.
    if min(precisions) > 0:
        log_mean = sum(math.log(p) for p in precisions) / max_n
        geo_mean = math.exp(log_mean)
    else:
        geo_mean = 0.0

    # Brevity penalty: BLEU is precision-only, so a one-word output could score
    # 1.0. The penalty is the counterweight against short outputs.
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return {
        "bleu": 100.0 * bp * geo_mean,
        "precisions": [100.0 * p for p in precisions],
        "brevity_penalty": bp,
        "hyp_len": hyp_len,
        "ref_len": ref_len,
    }


# ---------------------------------------------------------------- chrF

def corpus_chrf(hypotheses: list[str], references: list[str],
                max_n: int = 6, beta: float = 2.0) -> dict:
    """chrF: F-score over character n-grams, recall weighted beta times.

    Operates on raw character sequences with whitespace removed, so it is
    unaffected by tokenization and by how the two languages segment words -
    which is exactly why it is the most useful of the three metrics for
    Devanagari.
    """
    tp = [0] * max_n     # matched character n-grams
    hyp_tot = [0] * max_n
    ref_tot = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        h_chars = list(hyp.replace(" ", ""))
        r_chars = list(ref.replace(" ", ""))
        for n in range(1, max_n + 1):
            h_counts = ngrams(h_chars, n)
            r_counts = ngrams(r_chars, n)
            tp[n - 1] += sum(min(c, r_counts[g]) for g, c in h_counts.items())
            hyp_tot[n - 1] += sum(h_counts.values())
            ref_tot[n - 1] += sum(r_counts.values())

    precisions, recalls = [], []
    for i in range(max_n):
        precisions.append(tp[i] / hyp_tot[i] if hyp_tot[i] else 0.0)
        recalls.append(tp[i] / ref_tot[i] if ref_tot[i] else 0.0)

    # Average over orders first, then combine - this is the chrF definition,
    # and it differs from combining per order and then averaging.
    p = sum(precisions) / max_n
    r = sum(recalls) / max_n
    b2 = beta ** 2
    f = ((1 + b2) * p * r / (b2 * p + r)) if (p + r) > 0 else 0.0

    return {"chrf": 100.0 * f, "precision": 100.0 * p, "recall": 100.0 * r}


# ------------------------------------------------------------- ROUGE-L

def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time.

    Two rolling rows rather than a full table: the continuations here are a few
    hundred tokens, so the full matrix would be fine, but the rolling version is
    the same code and does not grow with corpus size.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def corpus_rouge_l(hypotheses: list[list[str]], references: list[list[str]],
                   beta: float = 1.2) -> dict:
    """ROUGE-L, aggregated over the corpus.

    Precision is LCS / hypothesis length, recall is LCS / reference length, and
    the two are combined into an F-measure. beta = 1.2 is the value used in the
    original ROUGE package, weighting recall slightly above precision.
    """
    lcs_total = hyp_total = ref_total = 0
    for hyp, ref in zip(hypotheses, references):
        lcs_total += lcs_length(hyp, ref)
        hyp_total += len(hyp)
        ref_total += len(ref)

    p = lcs_total / hyp_total if hyp_total else 0.0
    r = lcs_total / ref_total if ref_total else 0.0
    b2 = beta ** 2
    f = ((1 + b2) * p * r / (r + b2 * p)) if (p + r) > 0 else 0.0

    return {"rouge_l": 100.0 * f, "precision": 100.0 * p, "recall": 100.0 * r}


# ------------------------------------------------- diversity / degeneration

def diversity_stats(token_lists: list[list[str]]) -> dict:
    """Distinct-1, Distinct-2 and repetition rate over generated text.

    These detect the characteristic failure of a small language model, which is
    not incoherence but *degeneration*: falling into a loop and emitting the same
    phrase until the token budget runs out. Perplexity does not see this at all -
    a repeated high-probability phrase has excellent perplexity - so a diversity
    diagnostic is the thing that catches it.

    Distinct-n = unique n-grams / total n-grams, pooled across all generations.
    Values near 1 mean the model rarely repeats itself; values near 0 mean it is
    looping.

    repetition_rate = the share of 4-gram *occurrences* that are not the first
    appearance of that 4-gram. Fluent text repeats some 4-grams, so a low
    non-zero value is normal; above roughly 0.5 the output is mostly loop.
    """
    all_uni, all_bi, all_four = Counter(), Counter(), Counter()
    n_tokens = 0
    for tokens in token_lists:
        n_tokens += len(tokens)
        all_uni.update(ngrams(tokens, 1))
        all_bi.update(ngrams(tokens, 2))
        all_four.update(ngrams(tokens, 4))

    total_uni = sum(all_uni.values())
    total_bi = sum(all_bi.values())
    total_four = sum(all_four.values())
    repeated_four = sum(c - 1 for c in all_four.values() if c > 1)

    return {
        "tokens_generated": n_tokens,
        "distinct_1": len(all_uni) / total_uni if total_uni else 0.0,
        "distinct_2": len(all_bi) / total_bi if total_bi else 0.0,
        "repetition_rate_4gram": repeated_four / total_four if total_four else 0.0,
        "unique_unigrams": len(all_uni),
        "unique_bigrams": len(all_bi),
    }


# --------------------------------------------------------- bits per byte

def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert summed negative log-likelihood in nats to bits per UTF-8 byte.

    This is the metric that makes Model H and Model L comparable, and the reason
    the specification asks for it alongside perplexity.

    Perplexity is per *token*, and the two models tokenize differently - Marathi
    at 2.6301 tokens per word, Konkani at 2.5279. A model whose tokenizer splits
    text into more, shorter pieces has an easier per-token prediction problem and
    a lower perplexity, without being a better model of the language. Dividing
    the same total likelihood by the number of *bytes* removes the tokenizer from
    the denominator entirely: bytes are a property of the text, not of how it was
    segmented.

        BPB = (NLL in nats / ln 2) / bytes

    Lower is better, and the two numbers can be placed side by side.
    """
    return (total_nll_nats / math.log(2)) / total_bytes if total_bytes else 0.0


if __name__ == "__main__":
    # Sanity checks with hand-computable answers. A metric implementation that
    # has never been tested against a known value is a guess.
    ident = [["a", "b", "c", "d", "e", "f"]]
    print("identical BLEU (expect 100):",
          f"{corpus_bleu(ident, ident)['bleu']:.1f}")
    print("identical chrF (expect 100):",
          f"{corpus_chrf(['abcdef'], ['abcdef'])['chrf']:.1f}")
    print("identical ROUGE-L (expect 100):",
          f"{corpus_rouge_l(ident, ident)['rouge_l']:.1f}")

    disjoint = [["x", "y", "z", "w", "v", "u"]]
    print("disjoint BLEU (expect 0):",
          f"{corpus_bleu(disjoint, ident)['bleu']:.1f}")
    print("disjoint ROUGE-L (expect 0):",
          f"{corpus_rouge_l(disjoint, ident)['rouge_l']:.1f}")

    print("LCS('abcde','ace') (expect 3):",
          lcs_length(list("abcde"), list("ace")))

    loop = [["a", "b", "c", "d"] * 10]
    d = diversity_stats(loop)
    print(f"looping text: distinct_1 {d['distinct_1']:.3f} "
          f"repetition {d['repetition_rate_4gram']:.3f}  (expect low / high)")
