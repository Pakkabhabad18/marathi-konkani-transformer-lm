#!/usr/bin/env python3
"""
Konkani source K-9: MACHINE-TRANSLATED Marathi -> Konkani (D-036).

AUTHORISATION AND ITS CONDITIONS
--------------------------------
Synthetic/MT data was NOT permitted at the start of this project. It was
authorised by the TAs on 18 Aug 2026, with three conditions attached, quoted:

    "You may use this if and only if you cannot reach 500M after collecting
     all available real datasets (whether 80% or 20%)."
    "If you use it, you must justify your reasoning (what went wrong, what
     manual methods you tried & how you exhausted real data sources)."
    "MT or synthetic data should be your last resort. And you will be
     penalized if there were real datasets for your language."

Condition 2 is a deliverable, not a formality: `report/phase1_konkani_mt.md`
records what this run produced and why it was needed. Condition 3 is why every
real source is extracted first - the penalty is for reaching for MT while real
data was still available, not for using MT.

WHY MARATHI IS THE SOURCE
-------------------------
Marathi is the closest major language to Konkani, so it is the pair where MT
quality is highest; we already hold 177M manually collected Marathi words, so
the source text costs nothing to obtain and its provenance is already
documented. Translating from English or Hindi would be lower quality into
Konkani and more obviously translationese under inspection.

THE FAILURE MODE THIS SCRIPT GUARDS AGAINST
-------------------------------------------
Neural MT into a low-resource target very often COPIES the source instead of
translating it. Marathi and Konkani share Devanagari and much vocabulary, so a
copied Marathi sentence looks superficially fine and would silently inject
Marathi into the Konkani corpus - the exact contamination the whole project is
built to prevent, and worse here because Model H's training data would be
leaking into Model L's.

Two independent guards run on every output:

    1. COPY DETECTION - token-level Jaccard similarity against the source
       sentence. Above `--copy-threshold` the output is discarded as untranslated.
    2. LANGUAGE GATE - the same closed-class discriminator every other Konkani
       source passes through. Output labelled `mr` is discarded.

Both rejection counts are reported. If they dominate, the model is not actually
translating and the run should be abandoned rather than shipped - that verdict
is measured, not assumed.

CHECKPOINTING
-------------
The run is designed to be stopped at any moment and still be useful:

    - Ctrl-C (SIGINT) or SIGTERM sets a flag; the current batch finishes, the
      shard and manifest are flushed, and the checkpoint is written.
    - The checkpoint stores the source shard and the line offset within it, so
      a re-run resumes exactly where it stopped and never re-translates.
    - `--max-hours` self-stops cleanly, for leaving it running unattended.
    - Everything written before the stop is complete and usable; there is no
      "all or nothing" batch at the end.

CLASSIFICATION
--------------
`CollectionType.MACHINE_TRANSLATED`. Not manual (it cannot count toward the 20%
floor) and deliberately NOT folded into `downloaded_dataset` either, so that
every report can state the synthetic share separately.

USAGE
-----
    python3 konkani/scripts/generate_mt_konkani.py --pilot 200
    python3 konkani/scripts/generate_mt_konkani.py --max-hours 9.5
    python3 konkani/scripts/generate_mt_konkani.py            # until Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.dedup import Deduplicator, exact_hash                     # noqa: E402
from common.manifest import (                                         # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

SOURCE_NAME = "mt_marathi_to_konkani_indictrans2"
LANGUAGE = "kok"

DATA_DIR = REPO_ROOT / "konkani" / "data"
OUT_DIR = DATA_DIR / "synthetic" / SOURCE_NAME
MANIFEST_PATH = DATA_DIR / "manifests" / f"{SOURCE_NAME}.jsonl"
STATE_PATH = DATA_DIR / "checkpoints" / f"{SOURCE_NAME}_state.json"

# Source text: our own manually collected Marathi. Directories, in order.
SOURCE_DIRS = [
    REPO_ROOT / "marathi" / "data" / "manual" / "archive_org_maharashtra_gr",
    REPO_ROOT / "marathi" / "data" / "manual" / "news",
]

MIN_WORDS = 25
MIN_DEVANAGARI_RATIO = 0.70
TARGET_DOC_WORDS = 300
SHARD_SIZE = 5000
MIN_SENT_WORDS = 4
MAX_SENT_WORDS = 60

# Decoding length cap. Generation cost is roughly linear in max_length (and
# QUADRATIC when the KV cache is unavailable), so this is the single biggest
# throughput lever available on CPU. Source sentences are capped at
# MAX_SENT_WORDS = 60 words; at IndicTrans2's ~2 tokens/word for Devanagari
# that is ~120 tokens, so 160 leaves headroom for expansion during translation
# while cutting the old 256 default by 37%. Raising it only buys the ability to
# translate sentences we never feed it.
MAX_GEN_TOKENS = 160

_SENT_SPLIT = re.compile(r"(?<=[।?!])\s+")
_WORD_RE = re.compile(r"[ऀ-ॿ]+")

_stop = False


def _handle_stop(signum, frame):        # noqa: ARG001
    global _stop
    if _stop:
        print("\n  Second interrupt - exiting immediately.", flush=True)
        raise KeyboardInterrupt
    _stop = True
    print("\n  Stop requested. Finishing the current batch, then flushing "
          "and checkpointing...", flush=True)


class Engine:
    """Translation backend. IndicTrans2 preferred, NLLB as fallback."""

    def __init__(self, name: str, batch_size: int, prefer_device: str = "auto"):
        self.name = name
        self.batch_size = batch_size
        self.prefer_device = prefer_device
        self.device = "cpu"
        self._no_cache = False        # flipped by the D-041 compat fallback
        if name == "indictrans2":
            self._init_indictrans2()
        elif name == "nllb":
            self._init_nllb()
        else:
            raise SystemExit(f"unknown engine {name!r}")

    def _init_indictrans2(self):
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            try:
                from IndicTransToolkit import IndicProcessor        # older layout
            except ImportError:
                raise SystemExit(
                    "IndicTrans2 needs IndicTransToolkit:\n"
                    "    pip install "
                    "git+https://github.com/VarunGumma/IndicTransToolkit.git\n"
                    "Or run with --engine nllb, which needs only transformers.")
        import os
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # PyTorch defaults to a conservative thread count. This is a pure-CPU
        # workload on a multi-core laptop, so use every core.
        cores = os.cpu_count() or 4
        torch.set_num_threads(cores)

        # DEVICE SELECTION (D-042)
        # Measured on this machine: 0.9 sentences/s on CPU, which yields ~2.2M
        # words over two days - about 4% of what is needed. Apple Silicon has a
        # GPU reachable through torch's "mps" backend, so it is tried first.
        # IndicTrans2 ships custom modeling code that is not guaranteed to be
        # MPS-clean, so this is attempted and VERIFIED with a real forward pass
        # rather than assumed: if anything raises, we fall back to CPU and say
        # so. A silent fallback would look like "the GPU didn't help".
        self.device = "cpu"
        if self.prefer_device in ("auto", "mps") and \
                getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            self.device = "mps"
        elif self.prefer_device == "cpu":
            self.device = "cpu"
        print(f"  torch threads: {cores}   device: {self.device}", flush=True)

        model_id = "ai4bharat/indictrans2-indic-indic-dist-320M"
        print(f"  loading {model_id} ...", flush=True)
        self.processor = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32)

        # Passing use_cache=False to generate() is NOT enough on transformers
        # >= ~4.49: the cache object is built from the MODEL's config before
        # generate()'s argument is consulted, so IndicTrans2's legacy-format
        # decoder still receives a Cache it cannot index. Both flags below have
        # to be cleared at load time. If this still fails, the environment is
        # simply too new for IndicTrans2's vendored modeling code and the fix
        # is a pinned transformers (see the venv recipe in the run notes) -
        # there is no argument to generate() that repairs it.
        self.model.config.use_cache = False
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = False

        self.model.eval()

        if self.device == "mps":
            try:
                self.model = self.model.to("mps")
                probe = self.tokenizer(
                    self.processor.preprocess_batch(
                        ["हे एक चाचणी वाक्य आहे."],
                        src_lang="mar_Deva", tgt_lang="gom_Deva"),
                    return_tensors="pt", return_attention_mask=True,
                    padding="longest", truncation=True, max_length=MAX_GEN_TOKENS)
                probe = {k: v.to("mps") for k, v in probe.items()}
                with torch.no_grad():
                    self.model.generate(**probe, num_beams=1, max_length=32)
                print("  MPS probe OK - running on the GPU.", flush=True)
            except Exception as exc:                            # noqa: BLE001
                print(f"  MPS probe FAILED ({exc.__class__.__name__}: {exc}); "
                      f"falling back to CPU.", flush=True)
                self.device = "cpu"
                self.model = self.model.to("cpu")

        self.torch = torch
        self.src, self.tgt = "mar_Deva", "gom_Deva"

    def _init_nllb(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = "facebook/nllb-200-distilled-600M"
        print(f"  loading {model_id} ...", flush=True)
        self.processor = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       src_lang="mar_Deva")
        self.src, self.tgt = "mar_Deva", "gom_Deva"

        # NLLB-200 DOES NOT SUPPORT KONKANI. Verified against the model's
        # special_tokens_map.json on 19 Aug 2026: no `gom_Deva`, no `kok_*`.
        #
        # This check exists because the failure is silent without it.
        # `convert_tokens_to_ids` returns the UNKNOWN id for a language code
        # the model has never seen, `generate` accepts that id as
        # forced_bos_token_id without complaint, and the model emits fluent
        # text in some other language. Marathi and Hindi share Devanagari with
        # Konkani, so the output would pass the script check, partly survive
        # the language gate, and land in the corpus as counterfeit Konkani.
        # Refusing loudly is the only safe behaviour.
        tgt_id = self.tokenizer.convert_tokens_to_ids(self.tgt)
        unk_id = self.tokenizer.unk_token_id
        if tgt_id is None or tgt_id == unk_id:
            raise SystemExit(
                f"{model_id} has no '{self.tgt}' language token - it does not "
                f"support Konkani, and would silently generate a DIFFERENT "
                f"language.\n"
                f"Use --engine indictrans2 instead. Its repo is gated 'auto' "
                f"(self-serve):\n"
                f"  1. sign in at https://huggingface.co\n"
                f"  2. open https://huggingface.co/ai4bharat/"
                f"indictrans2-indic-indic-dist-320M and accept the terms\n"
                f"  3. hf auth login   (paste a read token)\n")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        self.torch = torch

    def _finish(self, out):
        """Decode + detokenise a generate() result."""
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return self.processor.postprocess_batch(decoded, lang=self.tgt)

    def translate(self, sentences: list[str]) -> list[str]:
        torch = self.torch
        if self.processor is not None:
            prepared = self.processor.preprocess_batch(
                sentences, src_lang=self.src, tgt_lang=self.tgt)
            # return_attention_mask=True is REQUIRED and is not the default for
            # IndicTrans2's custom tokenizer. Without it the tokenizer returns
            # no `attention_mask`, IndicTrans2's generate() reads
            # attention_mask.shape unconditionally, and every batch dies with
            # "'NoneType' object has no attribute 'shape'". This is the
            # documented call signature from the model card, not a workaround.
            batch = self.tokenizer(
                prepared,
                truncation=True,
                padding="longest",
                max_length=MAX_GEN_TOKENS,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if self.device != "cpu":
                batch = {k: v.to(self.device) for k, v in batch.items()}
            # KV-CACHE COMPATIBILITY (D-041)
            # -------------------------------
            # IndicTrans2 ships custom modeling code written against the
            # legacy transformers cache format (a tuple of tuples). Its
            # decoder does:
            #
            #     past_key_values[0][0].shape[2] if past_key_values is not None else 0
            #
            # transformers >= ~4.49 passes a `Cache` OBJECT instead. That
            # object is not None, so the guard passes, but indexing an
            # empty cache yields None -> "'NoneType' object has no
            # attribute 'shape'" on the very first step.
            #
            # Disabling the cache makes past_key_values genuinely None, so
            # the `else 0` branch runs and generation works on any
            # transformers version. The cost is real: without a KV cache
            # each step recomputes the whole prefix, so decoding is several
            # times slower. Pinning transformers to a 4.4x release would
            # restore the fast path, which is why this is a runtime FALLBACK
            # rather than a hard-coded setting - if the fast path works, we
            # keep it.
            with torch.no_grad():
                if not self._no_cache:
                    try:
                        return self._finish(self.model.generate(
                            **batch, num_beams=1, min_length=0,
                            max_length=MAX_GEN_TOKENS, use_cache=True))
                    except AttributeError as exc:
                        if "shape" not in str(exc):
                            raise
                        self._no_cache = True
                        print("  [compat] transformers cache format is "
                              "incompatible with IndicTrans2's custom modeling "
                              "code; falling back to use_cache=False for the "
                              "rest of the run (slower but correct).",
                              flush=True)
                out = self.model.generate(
                    **batch,
                    num_beams=1,          # greedy: this runs on CPU
                    min_length=0,
                    max_length=MAX_GEN_TOKENS,
                    use_cache=False,
                )
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            return self.processor.postprocess_batch(decoded, lang=self.tgt)

        batch = self.tokenizer(sentences, truncation=True, padding=True,
                               max_length=256, return_tensors="pt")
        forced = self.tokenizer.convert_tokens_to_ids(self.tgt)
        with torch.no_grad():
            out = self.model.generate(**batch, forced_bos_token_id=forced,
                                      num_beams=1, max_length=256)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


def jaccard(a: str, b: str) -> float:
    """Token-level overlap. 1.0 means the output is a copy of the input."""
    sa, sb = set(_WORD_RE.findall(a)), set(_WORD_RE.findall(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def iter_source_sentences(state: dict):
    """Yield Marathi sentences, resuming from the recorded position."""
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        if directory.exists():
            files.extend(sorted(directory.rglob("shard_*.txt")))
    if not files:
        raise SystemExit(f"no Marathi source shards under {SOURCE_DIRS}")

    done_files = set(state.get("completed_files", []))
    resume_file = state.get("current_file")
    resume_line = int(state.get("current_line", 0))

    for path in files:
        key = str(path.relative_to(REPO_ROOT))
        if key in done_files:
            continue
        start_line = resume_line if key == resume_file else 0
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"    [skip] {path.name}: {exc}")
            continue
        with handle:
            for lineno, line in enumerate(handle):
                if lineno < start_line:
                    continue
                for sent in _SENT_SPLIT.split(line.strip()):
                    sent = sent.strip()
                    n = len(sent.split())
                    if MIN_SENT_WORDS <= n <= MAX_SENT_WORDS:
                        yield key, lineno, sent
        done_files.add(key)
        state["completed_files"] = sorted(done_files)
        state["current_file"] = None
        state["current_line"] = 0


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print("  checkpoint unreadable; starting fresh.")
    return {"completed_files": [], "current_file": None, "current_line": 0,
            "sentences_translated": 0, "documents_written": 0, "words": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)            # atomic; never a half-written checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate MT Konkani from our manual Marathi corpus.")
    ap.add_argument("--engine", choices=["indictrans2", "nllb"],
                    default="indictrans2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                    help="auto tries Apple GPU (mps) and falls back to cpu")
    ap.add_argument("--pilot", type=int, default=0,
                    help="translate only N sentences and report; writes nothing")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly after this many hours")
    ap.add_argument("--copy-threshold", type=float, default=0.90,
                    help="discard output this token-similar to its source")
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    args = ap.parse_args()

    dry = args.pilot > 0
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print("=" * 74)
    print(f"GENERATING (MACHINE-TRANSLATED): {SOURCE_NAME}")
    print("  direction : Marathi (our manual corpus) -> Konkani")
    print(f"  engine    : {args.engine}")
    print(f"  mode      : {'PILOT (writes nothing)' if dry else 'WRITE'}")
    if args.max_hours:
        print(f"  stop after: {args.max_hours:.2f} h")
    print("  NOTE: classified MACHINE_TRANSLATED - never counts as manual,")
    print("        and reported separately from downloaded text.")
    print("=" * 74)

    state = {"completed_files": [], "current_file": None, "current_line": 0,
             "sentences_translated": 0, "documents_written": 0, "words": 0} \
        if dry else load_state()
    if not dry and state.get("sentences_translated"):
        print(f"\n  RESUMING: {state['sentences_translated']:,} sentences and "
              f"{state['words']:,} words already done.")
        print(f"  completed source shards: {len(state['completed_files'])}")

    engine = Engine(args.engine, args.batch_size, args.device)
    print("  model ready.\n", flush=True)

    if not dry:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if dry else ManifestWriter(MANIFEST_PATH)

    seen_exact: set[str] = set()
    rejected: dict[str, int] = {}
    accepted = state.get("documents_written", 0)
    words_total = state.get("words", 0)
    sent_done = state.get("sentences_translated", 0)
    session_sentences = 0

    shard_index = len(sorted(OUT_DIR.glob("shard_*.txt"))) if not dry else 0
    shard = None
    if not dry:
        shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                     encoding="utf-8")

    doc_buf: list[str] = []
    doc_words = 0
    batch: list[tuple[str, int, str]] = []
    start = time.time()
    deadline = start + args.max_hours * 3600 if args.max_hours else None

    def flush_document():
        nonlocal doc_buf, doc_words, accepted, words_total, shard, shard_index
        if not doc_buf:
            return
        raw = " ".join(doc_buf)
        doc_buf, doc_words = [], 0
        text = normalize(raw, keep_paragraphs=False)
        if len(text.split()) < MIN_WORDS:
            rejected["doc_too_short"] = rejected.get("doc_too_short", 0) + 1
            return
        profile = profile_script(text)
        if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
            rejected["doc_not_devanagari"] = rejected.get("doc_not_devanagari", 0) + 1
            return
        langid = identify_marathi_konkani(text)
        if langid.label == "mr":
            rejected["doc_langid_marathi"] = rejected.get("doc_langid_marathi", 0) + 1
            return
        h = exact_hash(text)
        if h in seen_exact:
            rejected["doc_exact_duplicate"] = rejected.get("doc_exact_duplicate", 0) + 1
            return
        seen_exact.add(h)
        if deduper.is_duplicate(text):
            rejected["doc_near_duplicate"] = rejected.get("doc_near_duplicate", 0) + 1
            return

        accepted += 1
        words_total += len(text.split())
        if manifest:
            manifest.write(make_record(
                text=text, raw_text=raw,
                source_name=SOURCE_NAME,
                source_url="local:marathi_manual_corpus",
                collection_type=CollectionType.MACHINE_TRANSLATED,
                language=LANGUAGE,
                preprocessing_applied=NORMALIZATION_STEPS + [
                    f"mt_{args.engine}_mar_to_gom", "copy_detection",
                    "langid_marathi_rejected", "exact_dedup", "near_dedup"],
                script=profile.script,
                langid_score=langid.score,
                langid_label=langid.label,
                devanagari_ratio=profile.devanagari_ratio,
                doc_id=f"{SOURCE_NAME}_{accepted:08d}",
                notes="SYNTHETIC: machine-translated from our manual Marathi "
                      "corpus. Authorised by TAs 18 Aug 2026 as last resort.",
            ))
        if shard:
            shard.write(text.replace("\n", " ") + "\n")
            if accepted % SHARD_SIZE == 0:
                shard.close()
                shard_index += 1
                shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                             encoding="utf-8")

    def run_batch():
        nonlocal doc_buf, doc_words, sent_done, session_sentences
        if not batch:
            return
        sources = [s for _, _, s in batch]
        try:
            outputs = engine.translate(sources)
        except Exception as exc:                                # noqa: BLE001
            rejected["batch_failed"] = rejected.get("batch_failed", 0) + len(batch)
            # If the very FIRST batch fails, the problem is configuration, not
            # data: every subsequent batch will fail identically. Printing the
            # same line 10,000 times and finishing with zero output wastes the
            # run and hides the traceback. Abort on the first failure and show
            # it; tolerate later failures, which are genuinely per-batch.
            if session_sentences == 0:
                import traceback
                print("\n  FIRST BATCH FAILED - aborting rather than repeating "
                      "this error for the whole run.\n")
                traceback.print_exc()
                raise SystemExit(
                    f"Translation failed on the first batch: "
                    f"{exc.__class__.__name__}: {exc}")
            print(f"    [batch error] {exc.__class__.__name__}: {exc}")
            batch.clear()
            return
        for (key, lineno, src), out in zip(batch, outputs):
            sent_done += 1
            session_sentences += 1
            state["current_file"] = key
            state["current_line"] = lineno
            out = (out or "").strip()
            if not out:
                rejected["empty_output"] = rejected.get("empty_output", 0) + 1
                continue
            if jaccard(src, out) >= args.copy_threshold:
                rejected["copied_not_translated"] = \
                    rejected.get("copied_not_translated", 0) + 1
                continue
            doc_buf.append(out)
            doc_words += len(out.split())
            if doc_words >= TARGET_DOC_WORDS:
                flush_document()
        batch.clear()

    try:
        for key, lineno, sent in iter_source_sentences(state):
            if _stop:
                break
            if deadline and time.time() >= deadline:
                print(f"\n  --max-hours reached; stopping cleanly.")
                break
            if args.pilot and session_sentences >= args.pilot:
                break

            batch.append((key, lineno, sent))
            if len(batch) >= args.batch_size:
                run_batch()
                if not dry and session_sentences % 2000 < args.batch_size:
                    if shard:
                        shard.flush()
                    if manifest:
                        manifest.flush() if hasattr(manifest, "flush") else None
                    state["sentences_translated"] = sent_done
                    state["documents_written"] = accepted
                    state["words"] = words_total
                    save_state(state)
                    el = time.time() - start
                    rate = session_sentences / max(el, 1e-9)
                    print(f"  sent={sent_done:,} docs={accepted:,} "
                          f"words={words_total:,} "
                          f"({rate:.1f} sent/s, {rate*3600*13/1e6:.1f}M words/h "
                          f"projected)", flush=True)
        run_batch()
        flush_document()

    except KeyboardInterrupt:
        print("\n  Interrupted hard.")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()
        if not dry:
            state["sentences_translated"] = sent_done
            state["documents_written"] = accepted
            state["words"] = words_total
            save_state(state)

    el = (time.time() - start) / 60
    print("\n" + "=" * 74)
    print("RUN SUMMARY")
    print("=" * 74)
    print(f"Sentences this session:  {session_sentences:,}")
    print(f"Sentences total:         {sent_done:,}")
    print(f"Documents accepted:      {accepted:,}")
    print(f"Words accepted:          {words_total:,}")
    print(f"Elapsed this session:    {el:.1f} min")
    if session_sentences and el:
        print(f"Throughput:              "
              f"{session_sentences / (el * 60):.1f} sentences/s")

    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<34}{count:>12,}")

    copied = rejected.get("copied_not_translated", 0)
    marathi = rejected.get("doc_langid_marathi", 0)
    if session_sentences and copied / max(session_sentences, 1) > 0.30:
        print("\n  WARNING: more than 30% of outputs were COPIES of the input.")
        print("  The model is not translating into Konkani. Do not ship this;")
        print("  try --engine nllb, or abandon MT and report the shortfall.")
    if marathi:
        print(f"\n  {marathi:,} assembled documents were still labelled Marathi")
        print("  after translation and were discarded.")

    print("\n" + "-" * 74)
    print("EFFECT ON THE KONKANI CORPUS")
    print("-" * 74)
    print(f"  synthetic words       {words_total:>15,}")
    print("  (MACHINE_TRANSLATED - not manual, and reported separately from")
    print("   downloaded text in every stats table)")
    if dry:
        print("\nPILOT: nothing was written.")
    else:
        print(f"\n  shards     {OUT_DIR.relative_to(REPO_ROOT)}")
        print(f"  manifest   {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        print(f"  checkpoint {STATE_PATH.relative_to(REPO_ROOT)}")
        print("\n  Safe to stop and resume at any time - re-run the same "
              "command to continue.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
