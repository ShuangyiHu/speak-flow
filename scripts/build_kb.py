#!/usr/bin/env python3
"""
scripts/build_kb.py
===================
One-time script to build the SpeakFlow debate knowledge base.

Downloads DebateSum from HuggingFace, chunks and filters the data,
scores argument quality, then indexes everything into ChromaDB via
RAGRetriever.index_chunks().

Usage:
    # From project root
    python scripts/build_kb.py

    # Limit rows for testing
    python scripts/build_kb.py --max-rows 500

    # Skip download if already cached
    python scripts/build_kb.py --skip-download

    # Wipe and rebuild from scratch
    python scripts/build_kb.py --reset

Dependencies (add to requirements.txt):
    datasets
    chromadb
    sentence-transformers
"""

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterator

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from scripts/ or from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rag_retriever import RAGRetriever, DebateChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# DebateSum HuggingFace dataset ID
DEBATESUM_DATASET = "Hellisotherpeople/DebateSum"

# Minimum word count for a chunk to be worth indexing
MIN_CHUNK_WORDS = 20
MAX_CHUNK_WORDS = 120

# How many abstract sentences to extract per document
SENTENCES_PER_DOC = 3

# Argument type classifiers — keyword-based heuristics
# Good enough for a first pass; can be replaced with a classifier later
COUNTER_KEYWORDS = [
    "however", "but", "on the other hand", "opponents argue",
    "critics claim", "in contrast", "yet", "nevertheless",
    "this ignores", "fails to account", "overlooks"
]
EVIDENCE_KEYWORDS = [
    "according to", "studies show", "research indicates", "data shows",
    "percent", "%", "million", "billion", "statistics", "survey",
    "published in", "journal", "reported that", "found that"
]
REBUTTAL_KEYWORDS = [
    "this argument fails", "the claim that", "while it may seem",
    "even if we accept", "the evidence contradicts", "this misunderstands"
]
FRAMEWORK_KEYWORDS = [
    "the key question", "we must first", "the principle",
    "at stake is", "the real issue", "fundamentally",
    "the framework", "the core tension"
]


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_abstract(abstract: str) -> list[str]:
    """
    Split a DebateSum abstract into overlapping sentence-window chunks.
    Each chunk is 2-3 consecutive sentences, shifted by 1 sentence.
    This preserves argument context better than fixed-character splits.
    """
    sentences = _split_sentences(abstract)
    if not sentences:
        return []

    chunks = []
    window = 2  # sentences per chunk

    for i in range(len(sentences)):
        chunk_sentences = sentences[i:i + window]
        chunk_text = " ".join(chunk_sentences)
        wc = _word_count(chunk_text)
        if MIN_CHUNK_WORDS <= wc <= MAX_CHUNK_WORDS:
            chunks.append(chunk_text)

    # If no chunks passed the filter, use the whole abstract trimmed
    if not chunks and _word_count(abstract) >= MIN_CHUNK_WORDS:
        words = abstract.split()
        chunks.append(" ".join(words[:MAX_CHUNK_WORDS]))

    return chunks


# ── Argument classification ───────────────────────────────────────────────────

def _classify_argument_type(text: str) -> str:
    """
    Heuristic keyword-based classifier for argument_type.
    Returns one of: "claim", "evidence", "counter_argument", "rebuttal", "framework"
    """
    lower = text.lower()

    # Check most specific first
    if any(kw in lower for kw in REBUTTAL_KEYWORDS):
        return "rebuttal"
    if any(kw in lower for kw in COUNTER_KEYWORDS):
        return "counter_argument"
    if any(kw in lower for kw in EVIDENCE_KEYWORDS):
        return "evidence"
    if any(kw in lower for kw in FRAMEWORK_KEYWORDS):
        return "framework"
    return "claim"  # default


def _score_strength(text: str, argument_type: str) -> float:
    """
    Heuristic strength scorer (0.0–1.0).
    Rewards: evidence presence, sufficient length, hedging avoidance.
    Penalizes: very short chunks, filler phrases.
    """
    score = 0.5  # baseline
    lower = text.lower()
    wc = _word_count(text)

    # Length bonus
    if wc >= 40:
        score += 0.1
    if wc >= 70:
        score += 0.05

    # Evidence signal bonus
    if any(kw in lower for kw in EVIDENCE_KEYWORDS):
        score += 0.15

    # Specific numbers/stats bonus
    if re.search(r'\d+(\.\d+)?%|\$\d+|\d+ (million|billion|thousand)', text):
        score += 0.1

    # Hedging penalty
    hedge_words = ["might", "perhaps", "possibly", "some argue", "it could be"]
    if sum(1 for h in hedge_words if h in lower) >= 2:
        score -= 0.1

    # Vague opener penalty
    if lower.startswith(("this is", "there are", "many people")):
        score -= 0.05

    # Evidence type gets inherent bonus
    if argument_type == "evidence":
        score += 0.1

    return round(max(0.1, min(1.0, score)), 2)


def _extract_topic(full_document: str) -> str:
    """
    Extract a short topic label from the full document text.
    DebateSum documents often start with the resolution — grab first 8 words.
    Falls back to "general_debate" if text is too short.
    """
    words = full_document.strip().split()
    if len(words) < 4:
        return "general_debate"
    # Take first 6 words, lowercase, remove punctuation
    topic = " ".join(words[:6]).lower()
    topic = re.sub(r'[^a-z0-9 ]', '', topic).strip()
    return topic[:80]  # cap at 80 chars


def _make_chunk_id(text: str, idx: int) -> str:
    """Stable, unique chunk ID based on content hash + index."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"debatesum_{h}_{idx}"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_debatesum(max_rows: int | None = None) -> Iterator[dict]:
    """
    Stream DebateSum from HuggingFace.
    Yields raw dataset rows.

    DebateSum schema:
        - 'Abstract'       : Extractive summary of the debate document (~100-300 words)
        - 'Full-Document'  : Full debate text (often very long, skip for chunking)
        - 'extract-abstractive-similarity' : float quality signal

    We use 'Abstract' for chunking — it's already condensed argument content.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Missing dependency: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading DebateSum from HuggingFace (streaming)...")
    ds = load_dataset(DEBATESUM_DATASET, split="train", streaming=True)

    count = 0
    for row in ds:
        if max_rows and count >= max_rows:
            break
        yield row
        count += 1
        if count % 500 == 0:
            logger.info(f"  Loaded {count} rows...")

    logger.info(f"Finished loading {count} rows from DebateSum")


# ── Main processing pipeline ──────────────────────────────────────────────────

def process_debatesum_row(row: dict, chunk_idx_start: int) -> list[DebateChunk]:
    """
    Convert one DebateSum row into a list of DebateChunk objects.
    Returns empty list if the row doesn't meet quality thresholds.
    """
    abstract = (row.get("Abstract") or "").strip()
    full_doc = (row.get("Full-Document") or "").strip()

    # Skip empty or very short abstracts
    if _word_count(abstract) < MIN_CHUNK_WORDS:
        return []

    topic = _extract_topic(full_doc or abstract)
    raw_chunks = _chunk_abstract(abstract)

    result = []
    for i, chunk_text in enumerate(raw_chunks):
        arg_type = _classify_argument_type(chunk_text)
        strength = _score_strength(chunk_text, arg_type)

        # Filter out low-quality chunks
        if strength < 0.4:
            continue

        chunk = DebateChunk(
            chunk_id=_make_chunk_id(chunk_text, chunk_idx_start + i),
            text=chunk_text,
            topic=topic,
            argument_type=arg_type,
            strength_score=strength,
            source="DebateSum",
            metadata={
                "similarity_score": row.get("extract-abstractive-similarity", 0.0),
            }
        )
        result.append(chunk)

    return result


async def build(args: argparse.Namespace) -> None:
    logger.info("=== SpeakFlow Knowledge Base Builder ===")

    # Wipe existing DB if requested
    if args.reset:
        import shutil
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        if Path(chroma_path).exists():
            shutil.rmtree(chroma_path)
            logger.info(f"Wiped existing ChromaDB at {chroma_path}")

    retriever = RAGRetriever()
    logger.info("RAGRetriever initialized")

    # ── Process DebateSum ──────────────────────────────────────────────────
    all_chunks: list[DebateChunk] = []
    chunk_idx = 0

    for row in load_debatesum(max_rows=args.max_rows):
        chunks = process_debatesum_row(row, chunk_idx)
        all_chunks.extend(chunks)
        chunk_idx += len(chunks)

    logger.info(f"Generated {len(all_chunks)} chunks from DebateSum")

    # ── Argument type distribution ─────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for c in all_chunks:
        type_counts[c.argument_type] = type_counts.get(c.argument_type, 0) + 1
    logger.info(f"Argument type distribution: {type_counts}")

    avg_strength = sum(c.strength_score for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"Average strength score: {avg_strength:.2f}")

    # ── Index into ChromaDB ────────────────────────────────────────────────
    logger.info("Indexing into ChromaDB...")
    indexed = await retriever.index_chunks(all_chunks)
    logger.info(f"✅ Successfully indexed {indexed} chunks")

    # ── Smoke test ────────────────────────────────────────────────────────
    logger.info("Running smoke test retrieval...")
    try:
        import dataclasses
        from shared_types import CoachingStrategy, CoachingAction, TurnAnalysis
        from shared_types import TurnInput, ArgumentResult, PronunciationResult
        from datetime import datetime

        dummy_action = CoachingAction(
            strategy=CoachingStrategy.PROBE,
            intent="User has a claim but no evidence",
            target_claim="renewable energy can solve climate change",
            target_word=None,
            target_phoneme=None,
            argument_score=0.45,
            pronunciation_score=0.8,
            difficulty_delta=0,
            turn_number=2,
            topic="renewable energy",
            user_position="for",
            prior_coach_responses=[]
        )

        # Build ArgumentResult dynamically — handles any extra fields
        # in shared_types without breaking this script
        arg_field_names = {f.name for f in dataclasses.fields(ArgumentResult)}
        arg_pool = {
            "has_claim": True,
            "has_reasoning": False,
            "has_evidence": False,
            "logical_gaps": ["No supporting data provided", "Causal mechanism unclear"],
            "vocabulary_flags": [],
            "argument_score": 0.45,
            "summary": "User asserts renewable energy is the future without reasoning.",
            "clarity_score": 0.5,
            "reasoning_score": 0.3,
            "depth_score": 0.3,
            "fluency_score_arg": 0.6,
            "clarity_feedback": "Claim is present but lacks specificity.",
            "reasoning_feedback": "No reasoning chain provided.",
            "depth_feedback": "Argument needs more development.",
            "fluency_feedback": "Expression is clear but basic.",
        }
        arg_kwargs = {k: v for k, v in arg_pool.items() if k in arg_field_names}
        dummy_argument = ArgumentResult(**arg_kwargs)

        dummy_analysis = TurnAnalysis(
            turn_input=TurnInput(
                transcript="I think renewable energy is the future.",
                session_id="test",
                turn_number=1,
                topic="renewable energy",
                user_position="for",
                audio_path="",
                prior_turns=[]
            ),
            argument=dummy_argument,
            pronunciation=PronunciationResult(
                mispronounced_words=[],
                fluency_score=0.8,
                target_phonemes=[]
            ),
            timestamp=datetime.now(),
            latency_ms=0
        )

        ctx = await retriever.retrieve(dummy_action, dummy_analysis, top_k=3)
        logger.info(f"Smoke test: retrieved {len(ctx.chunks)} chunks "
                    f"(fallback={ctx.fallback_used})")
        if ctx.chunks:
            logger.info(f"  Top chunk [{ctx.chunks[0].argument_type}, "
                        f"strength={ctx.chunks[0].strength_score}]: "
                        f"{ctx.chunks[0].text[:100]}...")
        logger.info(f"  HyDE query used: {ctx.hypothetical_query[:120]}...")

    except Exception as e:
        logger.warning(f"Smoke test failed (indexing still succeeded): {e}")

    logger.info("=== Build complete ===")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SpeakFlow debate knowledge base")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit number of DebateSum rows to process (default: all ~190k)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (use if already cached by HuggingFace)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe existing ChromaDB and rebuild from scratch"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(build(args))