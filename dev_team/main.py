#!/usr/bin/env python
"""
SpeakFlow AI — Dev Team Runner

Usage:
    python main.py                          # Run full pipeline for Turn Analyzer
    python main.py --module coach_policy    # Run for a different module
    python main.py --skip-github            # Run without GitHub PR integration

GitHub Integration:
    Set GITHUB_TOKEN and GITHUB_REPO env vars to enable automatic PR creation.
    The runner will:
      1. Syntax-check all generated Python files (abort if broken)
      2. Create a feature branch per module
      3. Upload generated files via GitHub API (no local git required)
      4. Open a PR with the review output as PR description
      5. Auto-merge if reviewer decision is APPROVE

Revision Loop:
    MAX_REVISIONS = 2  — max automated fix-and-re-review cycles
    Hard stop conditions (abort immediately, no further retries):
      - Same CRITICAL issue appears in two consecutive reviews
      - Generated file fails py_compile syntax check after revision
      - Revision engineer outputs empty or non-Python content
    On hard stop: pipeline aborts, files saved to output/ for manual inspection.
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase

from crew import SpeakFlowDevTeam
from github_integration import GitHubPRManager

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ── Revision loop configuration ───────────────────────────────────────────────
MAX_REVISIONS = 2  # max automated fix-and-re-review cycles (was 3)

# ── Module Definitions ────────────────────────────────────────────────────────

MODULES = {
    "turn_analyzer": {
        "module_name": "turn_analyzer.py",
        "class_name": "TurnAnalyzer",
        "requirements": """
The TurnAnalyzer module is the core real-time processing component of SpeakFlow AI,
an English debate coaching platform for Chinese L2 learners.

Responsibilities:
- Accept a spoken turn as input (transcript + audio path + session context)
- Analyze the argument structure using Claude (Claim-Reason-Evidence framework)
- Analyze pronunciation using Montreal Forced Aligner (MFA) phoneme alignment
- Return a structured TurnAnalysis object within 3 seconds

Data models required:
- TurnInput: transcript, session_id, turn_number, topic, user_position, audio_path, prior_turns
- ArgumentResult: has_claim, has_reasoning, has_evidence, logical_gaps, vocabulary_flags, argument_score, summary
- WordError: word, expected_ipa, actual_ipa, severity
- PronunciationResult: mispronounced_words, fluency_score, target_phonemes
- TurnAnalysis: turn_input, argument, pronunciation, timestamp, latency_ms

Key behaviors:
- Run argument analysis and pronunciation analysis concurrently (asyncio.gather)
- MFA call must use asyncio.to_thread (non-blocking)
- Include a stub MFA implementation toggled by USE_STUB_MFA env var
- Handle empty transcript gracefully without raising exceptions
- Use Anthropic Python SDK with model from os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
- Set timeout on AsyncAnthropic() init, NOT inside messages.create()
- Strip markdown fences before json.loads(): re.sub(r"```[a-z]*\\n?|```", "", text).strip()
- Access TurnAnalysis fields via analysis.argument.has_claim (NOT analysis.has_claim)

Acceptance criteria:
- analyze() returns TurnAnalysis within 3 seconds for 30-word input
- has_claim=True when transcript contains a clear position statement
- has_claim=False when transcript is a question or off-topic
- argument_score > 0.7 for well-structured CRE argument
- argument_score < 0.4 for opinion-only turn with no reasoning
- Empty transcript returns default TurnAnalysis, no exception raised
- mispronounced_words is empty list when USE_STUB_MFA=True (stub returns clean result)
""",
    },
    "coach_policy": {
        "module_name": "coach_policy.py",
        "class_name": "CoachPolicyAgent",
        "requirements": """
The CoachPolicyAgent decides the next coaching action based on TurnAnalysis output.
It is a LangGraph node that takes TurnAnalysis as state and returns a CoachingAction.

Responsibilities:
- Evaluate argument and pronunciation scores from TurnAnalysis
- Select a coaching strategy: PROBE, CHALLENGE, REDIRECT, PRAISE, CORRECT_PRONUNCIATION
- Generate a natural language response that feels like a debate partner, not a teacher
- Track coaching history to avoid repeating the same intervention twice in a row

Data models required:
- CoachingStrategy: Enum with PROBE, CHALLENGE, REDIRECT, PRAISE, CORRECT_PRONUNCIATION
- CoachingAction: strategy, response_text, target_skill, confidence_score
- CoachingState (LangGraph TypedDict): turn_analysis, coaching_history, session_id, turn_number

Key behaviors:
- _select_strategy(): uses analysis.argument.argument_score and analysis.argument.has_claim
  (NOT analysis.argument_score or analysis.has_claim — always go through analysis.argument)
- _generate_response(): calls Claude to generate natural debate-partner language
- _get_fallback_response(): returns hardcoded responses per strategy, no API call
- Use model from os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
- Set timeout on AsyncAnthropic() init, NOT inside messages.create()
- Strip markdown fences before json.loads(): re.sub(r"```[a-z]*\\n?|```", "", text).strip()
- Avoid repeating same strategy twice in a row (check coaching_history[-1] if exists)

Acceptance criteria:
- Returns CoachingAction within 3 seconds
- PRAISE when argument_score > 0.7
- PROBE when has_claim=True but has_evidence=False
- REDIRECT when argument_score < 0.3
- CORRECT_PRONUNCIATION when fluency_score < 0.6
- No exception raised on any valid TurnAnalysis input
- Fallback responses available for all 5 strategies (no API call required)
""",
    },
    "response_generator": {
        "module_name": "response_generator.py",
        "class_name": "ResponseGenerator",
        "requirements": """
The ResponseGenerator decouples natural language generation from CoachPolicyAgent decisions.
CoachPolicyAgent decides WHAT to do; ResponseGenerator decides HOW to say it.

Responsibilities:
- Take a CoachingAction (strategy + target_skill + confidence_score) and session context
- Generate a natural, encouraging debate-partner response in English
- Vary phrasing across turns to avoid repetitive feedback
- Support tone modes: SOCRATIC (questioning), CHALLENGING (pushback), AFFIRMING (praise)

Data models required:
- ResponseRequest: coaching_action, topic, user_position, prior_responses, turn_number
- GeneratedResponse: text, tone, follow_up_prompt, estimated_speaking_seconds

Key behaviors:
- Map CoachingStrategy → tone automatically
- Use Claude to generate varied, natural-sounding responses
- Keep responses to 1-2 sentences (debate partner, not lecturer)
- prior_responses: last 3 response texts, used to avoid repetition
- Model from os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
- Set timeout on AsyncAnthropic() init only
- Strip markdown fences before json.loads()

Acceptance criteria:
- Returns GeneratedResponse within 2 seconds
- Response text is 1-2 sentences, under 50 words
- Tone matches CoachingStrategy mapping
- No two consecutive responses are identical
- Fallback response available for all strategies (no API required)
""",
    },
    "session_evaluator": {
        "module_name": "session_evaluator.py",
        "class_name": "SessionEvaluator",
        "requirements": """
The SessionEvaluator module generates a structured end-of-session coaching report
for SpeakFlow AI, an English debate coaching platform for Chinese L2 learners.

It is called once per debate session, after all turns have completed.
It receives the full session history and uses Claude to produce a rich,
personalised report across five dimensions.

────────────────────────────────────────────────────────────────────────
IMPORTS — use shared_types.py for all shared models
────────────────────────────────────────────────────────────────────────
from shared_types import (
    SessionTurn, SessionMetadata,
    CoachingStrategy, ArgumentResult, PronunciationResult,
)

Do NOT redefine TurnAnalysis, CoachingAction, SessionTurn, etc.
Import them from shared_types. Only define new types in this module.

────────────────────────────────────────────────────────────────────────
DATA MODELS (new, defined in this module)
────────────────────────────────────────────────────────────────────────

@dataclass
class ArgumentProgressReport:
    opening_score: float            # argument_score of turn 1
    closing_score: float            # argument_score of last turn
    trend: str                      # "improving" | "declining" | "stable"
    best_turn_number: int
    best_turn_summary: str          # one sentence describing the best argument
    recurring_gaps: List[str]       # logical gaps that appeared in 3+ turns
    cre_completion_rate: float      # % of turns with all three: claim+reason+evidence

@dataclass
class PronunciationProgressReport:
    opening_fluency: float          # fluency_score of turn 1
    closing_fluency: float          # fluency_score of last turn
    trend: str                      # "improving" | "declining" | "stable"
    persistent_errors: List[str]    # words mispronounced in 3+ turns
    resolved_errors: List[str]      # words that were errors early but correct later
    target_phonemes: List[str]      # top 3 phonemes to practice (union of all turns)

@dataclass
class VocabularyReport:
    overused_words: List[str]       # words used 4+ times across turns
    strong_vocabulary: List[str]    # sophisticated words used correctly
    suggested_alternatives: dict    # overused_word -> [alternative1, alternative2]

@dataclass
class CoachingEffectivenessReport:
    strategy_counts: dict           # CoachingStrategy.value -> int
    most_used_strategy: str
    argument_response_to_probe: float    # avg score change after PROBE turns
    argument_response_to_challenge: float  # avg score change after CHALLENGE turns

@dataclass
class SessionReport:
    metadata: SessionMetadata
    argument_progress: ArgumentProgressReport
    pronunciation_progress: PronunciationProgressReport
    vocabulary: VocabularyReport
    coaching_effectiveness: CoachingEffectivenessReport
    overall_summary: str            # 2-3 sentence holistic summary (Claude-generated)
    top_strengths: List[str]        # 2 specific strengths, grounded in turn data
    top_improvements: List[str]     # 2 specific actionable improvements
    next_session_focus: str         # one concrete suggestion for next session
    timestamp: datetime
    latency_ms: int

────────────────────────────────────────────────────────────────────────
CORE CLASS
────────────────────────────────────────────────────────────────────────

class SessionEvaluator:

    def __init__(self, anthropic_api_key: str):
        Initialize with AsyncAnthropic client.
        Set timeout on AsyncAnthropic() init — NOT inside messages.create().
        Model: os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    async def evaluate(
        self,
        turns: List[SessionTurn],
        metadata: SessionMetadata,
    ) -> SessionReport:
        Main entry point. Returns SessionReport.
        - Raise ValueError if turns is empty.
        - Run _compute_stats() first (synchronous, no LLM).
        - Then call _generate_narrative() for the Claude-generated fields.
        - Combine into SessionReport.
        - Measure and record latency_ms.
        - On any exception: return _create_fallback_report(turns, metadata, start_time).

    def _compute_stats(
        self,
        turns: List[SessionTurn],
    ) -> dict:
        Pure computation — NO LLM calls. Returns a flat dict of all numeric
        and list-based stats that feed both the structured sub-reports and
        the Claude prompt. Keys include:
            argument_scores: List[float]
            fluency_scores: List[float]
            all_logical_gaps: List[str]          # flattened, with repetition
            all_mispronounced_words: List[str]   # flattened, with repetition
            all_vocabulary_flags: List[str]      # flattened, with repetition
            strategy_sequence: List[str]         # CoachingStrategy.value per turn
            cre_completions: int                 # turns where claim+reason+evidence all True
            total_turns: int

    async def _generate_narrative(
        self,
        turns: List[SessionTurn],
        stats: dict,
    ) -> dict:
        Single Claude API call. Returns a dict with keys:
            overall_summary: str
            top_strengths: List[str]       (exactly 2 items)
            top_improvements: List[str]    (exactly 2 items)
            next_session_focus: str
        Prompt must include: topic, user_position, total turns, argument trend,
        fluency trend, recurring gaps, persistent pronunciation errors.
        Instruct Claude to respond ONLY in JSON. No markdown fences.
        Strip markdown fences before json.loads():
            re.sub(r"```[a-z]*\n?|```", "", text).strip()
        On any parse failure: return _default_narrative().

    def _build_argument_progress(self, turns: List[SessionTurn], stats: dict) -> ArgumentProgressReport:
        Synchronous. Compute from stats dict. No LLM.
        trend logic: compare first-third vs last-third of argument_scores.
            improving if avg(last third) - avg(first third) > 0.1
            declining if avg(first third) - avg(last third) > 0.1
            stable otherwise
        recurring_gaps: gaps that appear in >= 3 turns (use Counter).

    def _build_pronunciation_progress(self, turns: List[SessionTurn], stats: dict) -> PronunciationProgressReport:
        Synchronous. Compute from stats dict. No LLM.
        persistent_errors: words mispronounced in >= 3 turns.
        resolved_errors: words mispronounced in turns 1-3 but NOT in last 3 turns.
        target_phonemes: union of target_phonemes from all PronunciationResults, top 3 by frequency.

    def _build_vocabulary_report(self, stats: dict) -> VocabularyReport:
        Synchronous. Compute from stats dict. No LLM.
        overused_words: vocabulary_flags appearing 4+ times (use Counter).
        strong_vocabulary: vocabulary_flags that appear exactly once
            AND word length > 6 (proxy for sophisticated vocabulary).
        suggested_alternatives: hardcoded dict for top 10 common overused words.
            e.g. {"think": ["argue", "contend", "maintain"],
                  "bad": ["detrimental", "harmful", "counterproductive"], ...}
            Only include keys that appear in overused_words.

    def _build_coaching_effectiveness(self, turns: List[SessionTurn], stats: dict) -> CoachingEffectivenessReport:
        Synchronous. Compute from stats dict. No LLM.
        For argument_response_to_probe and argument_response_to_challenge:
            find turns where that strategy was used, then compute
            avg(argument_score[turn+1] - argument_score[turn]) for those turns.
            Return 0.0 if no such turns exist.

    def _create_fallback_report(
        self,
        turns: List[SessionTurn],
        metadata: SessionMetadata,
        start_time: float,
    ) -> SessionReport:
        Synchronous. Returns a minimal valid SessionReport with:
        - All numeric scores computed from turns (no LLM)
        - overall_summary = "Session analysis unavailable."
        - top_strengths = [], top_improvements = [], next_session_focus = ""

    def _default_narrative(self) -> dict:
        Returns hardcoded fallback dict for _generate_narrative() parse failures.

────────────────────────────────────────────────────────────────────────
ANTHROPIC SDK PATTERN
────────────────────────────────────────────────────────────────────────
# In __init__:
self.anthropic_client = AsyncAnthropic(
    api_key=anthropic_api_key,
    timeout=float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30")),
)
self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# In _generate_narrative:
response = await self.anthropic_client.messages.create(
    model=self.model,
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}]
)
text = response.content[0].text
text = re.sub(r"```[a-z]*\n?|```", "", text).strip()
data = json.loads(text)

────────────────────────────────────────────────────────────────────────
ACCEPTANCE CRITERIA
────────────────────────────────────────────────────────────────────────
- evaluate() raises ValueError when turns=[]
- evaluate() returns SessionReport within 10 seconds for a 10-turn session
- argument_progress.trend == "improving" when scores go 0.3 → 0.8 across turns
- argument_progress.trend == "stable"    when scores stay within ±0.1
- pronunciation_progress.persistent_errors contains only words seen in 3+ turns
- vocabulary.overused_words contains only words flagged 4+ times
- on Anthropic API failure: returns fallback report, no exception raised
- overall_summary, top_strengths, top_improvements, next_session_focus
  are non-empty strings in the normal path
- latency_ms is recorded and present in the returned SessionReport
- Strip markdown fences before json.loads() — never call json.loads() on raw API text
""",
    },
    "rag_retriever": {
        "module_name": "rag_retriever.py",
        "class_name": "RAGRetriever",
        "requirements": """
The RAGRetriever module provides grounded debate knowledge retrieval for SpeakFlow AI.
It is called between CoachPolicyAgent and ResponseGenerator to enrich coaching responses
with concrete evidence, argument examples, and counter-argument patterns retrieved from
a curated debate knowledge base.

────────────────────────────────────────────────────────────────────────
DESIGN PHILOSOPHY
────────────────────────────────────────────────────────────────────────
The core insight: users' spoken arguments are often weak or vague. Retrieving
directly on a weak argument yields poor results. Instead, we use Hypothetical
Document Embedding (HyDE): first generate a hypothetical "ideal version" of the
user's argument using a lightweight LLM call, then use THAT as the retrieval query.
This dramatically improves retrieval relevance without requiring perfect user input.

Retrieval results are filtered by CoachingStrategy so that:
- PROBE/CHALLENGE → return counter-arguments and evidence patterns
- PRAISE → return examples of structurally strong arguments on the same topic
- REDIRECT → return topic-anchoring claims and framing examples
- CORRECT_PRONUNCIATION → skip retrieval (not applicable), return empty context

────────────────────────────────────────────────────────────────────────
IMPORTS — use shared_types.py for all shared models
────────────────────────────────────────────────────────────────────────
from shared_types import CoachingStrategy, CoachingAction, TurnAnalysis

Do NOT redefine CoachingStrategy or CoachingAction.
Import them from shared_types. Only define new types in this module.

────────────────────────────────────────────────────────────────────────
EXTERNAL DEPENDENCIES
────────────────────────────────────────────────────────────────────────
- chromadb                  — local vector store (pip install chromadb)
- sentence-transformers     — embedding model (pip install sentence-transformers)
  Use model: "all-MiniLM-L6-v2" (fast, good quality, 384-dim)
- anthropic (AsyncAnthropic) — for HyDE generation only
- Standard library: asyncio, os, json, re, dataclasses, typing, datetime, pathlib

────────────────────────────────────────────────────────────────────────
DATA MODELS (new, defined in this module)
────────────────────────────────────────────────────────────────────────

@dataclass
class DebateChunk:
    chunk_id: str
    text: str                       # The actual debate content
    topic: str                      # e.g. "renewable energy", "universal basic income"
    argument_type: str              # "claim", "evidence", "counter_argument", "rebuttal", "framework"
    strength_score: float           # 0.0–1.0, quality rating of this argument chunk
    source: str                     # Dataset origin: "DebateSum" | "IBM_ArgKP" | "manual"
    metadata: dict                  # Any additional fields (e.g. original doc_id, year)

@dataclass
class RetrievalContext:
    chunks: List[DebateChunk]       # Top-k retrieved and re-ranked chunks
    hypothetical_query: str         # The HyDE-generated query used for retrieval
    strategy_filter: str            # Which CoachingStrategy triggered this retrieval
    retrieval_latency_ms: int
    fallback_used: bool             # True if vector store unavailable or retrieval failed

────────────────────────────────────────────────────────────────────────
CORE CLASS
────────────────────────────────────────────────────────────────────────

class RAGRetriever:

    def __init__(self, collection_name: str = "speakflow_debate_kb"):
        Initialize:
        - Load SentenceTransformer("all-MiniLM-L6-v2") for embeddings
        - Connect to ChromaDB (persistent client, path from env var
          CHROMA_DB_PATH or default "./chroma_db")
        - Get or create collection with name collection_name
        - Initialize AsyncAnthropic client with timeout from
          os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30")
        - Set self._kb_available = True/False depending on whether
          ChromaDB collection has > 0 documents

    async def retrieve(
        self,
        coaching_action: CoachingAction,
        turn_analysis: TurnAnalysis,
        top_k: int = 3,
    ) -> RetrievalContext:
        Main entry point. Called by the LangGraph pipeline after CoachPolicyAgent.

        Steps:
        1. If strategy is CORRECT_PRONUNCIATION, return empty RetrievalContext
           immediately (retrieval not applicable for pronunciation coaching).
        2. If self._kb_available is False, return fallback RetrievalContext
           with hardcoded stub chunks (see _get_fallback_context).
        3. Call _generate_hypothetical_query(coaching_action, turn_analysis)
           to get a strong hypothetical argument for retrieval.
        4. Embed the hypothetical query using self._embed(text).
        5. Query ChromaDB with the embedding, applying metadata filter for
           strategy-appropriate argument_types (see _get_type_filter).
        6. Parse results into List[DebateChunk].
        7. Re-rank chunks with _rerank(chunks, coaching_action) —
           sort by: (0.6 * similarity_score + 0.4 * chunk.strength_score).
        8. Return RetrievalContext with top_k chunks.

        Must complete within 2 seconds total. Use asyncio.wait_for with timeout.
        On any exception, log the error and return _get_fallback_context().

    async def _generate_hypothetical_query(
        self,
        coaching_action: CoachingAction,
        turn_analysis: TurnAnalysis,
    ) -> str:
        HyDE: generate an ideal version of the user's argument for retrieval.

        IMPORTANT: turn_analysis MUST be used to extract the user's actual argument
        weakness. Specifically, read turn_analysis.argument.summary to get a
        one-sentence description of what the user actually said, and read
        turn_analysis.argument.logical_gaps (a List[str]) to get identified weaknesses.
        These must appear in the prompt so the hypothetical query is grounded in
        the user's real argument, not just the topic.

        Build the Claude prompt exactly like this (fill in the placeholders):

          prompt = (
              f"Write a 2-sentence strong debate argument on the topic: '{coaching_action.topic}'. "
              f"Position: {coaching_action.user_position}. "
              f"The student's current argument summary: '{turn_analysis.argument.summary}'. "
              f"Their identified weaknesses: {turn_analysis.argument.logical_gaps}. "
              f"Write the ideal version of this argument that fixes those weaknesses. "
              f"Return only the 2-sentence argument, no preamble."
          )

        Call Claude with that prompt. Use model from os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5").
        Use asyncio.wait_for with HYDE_TIMEOUT_SECONDS. On any failure or timeout,
        return coaching_action.topic as the fallback string.
        Strip markdown fences before returning: re.sub(r"```[a-z]*\n?|```", "", text).strip()

    def _embed(self, text: str) -> List[float]:
        Synchronous. Use self._embedding_model.encode(text).tolist().
        Returns a list of floats (384-dim vector for all-MiniLM-L6-v2).

    def _get_type_filter(self, strategy: CoachingStrategy) -> List[str]:
        Map CoachingStrategy to argument_types for ChromaDB metadata filtering.

        Mapping:
          PROBE             → ["evidence", "framework"]
          CHALLENGE         → ["counter_argument", "rebuttal"]
          REDIRECT          → ["claim", "framework"]
          PRAISE            → ["claim", "evidence"]   (show strong examples)
          CORRECT_PRONUNCIATION → []                  (never reaches here)

    def _rerank(
        self,
        chunks: List[DebateChunk],
        distances: List[float],
        coaching_action: CoachingAction,
    ) -> List[DebateChunk]:
        Synchronous. Re-rank retrieved chunks by composite score.
        composite = 0.6 * (1 - distance) + 0.4 * chunk.strength_score
        Sort descending. Return sorted list.

    def _get_fallback_context(self, strategy: CoachingStrategy) -> RetrievalContext:
        Return a hardcoded RetrievalContext with 2 stub DebateChunks
        appropriate for the given strategy. Used when KB is unavailable
        or retrieval fails. fallback_used must be True.

        Stub chunks must be non-empty, plausible debate content —
        not placeholder strings like "example argument here".
        Write real, useful stub content for at least:
          PROBE: evidence-seeking framework ("What data would your opponent demand...")
          CHALLENGE: a concrete counter-argument pattern
          REDIRECT: a topic-anchoring prompt

    async def index_chunks(self, chunks: List[DebateChunk]) -> int:
        Async wrapper for batch indexing. Used during knowledge base setup.
        This method MUST have a complete implementation body — not just a docstring.

        Implementation must do ALL of the following steps:
        1. Initialize a counter: indexed_count = 0
        2. Split chunks into batches of 50: use list slicing chunks[i:i+50]
        3. For each batch, iterate over each chunk and:
           a. Call embedding_vector = await asyncio.to_thread(self._embed, chunk.text)
           b. Collect: ids, embeddings, metadatas, documents lists for the batch
              - ids: [chunk.chunk_id]
              - embeddings: [embedding_vector]
              - documents: [chunk.text]
              - metadatas: [{"topic": chunk.topic, "argument_type": chunk.argument_type,
                             "strength_score": chunk.strength_score, "source": chunk.source}]
        4. Call self._collection.upsert(ids=ids, embeddings=embeddings,
                                         documents=documents, metadatas=metadatas)
        5. Add len(batch) to indexed_count
        6. After all batches, set self._kb_available = True
        7. Return indexed_count

        Wrap the entire method body in try/except Exception, log errors, return 0 on failure.

────────────────────────────────────────────────────────────────────────
CONSTANTS
────────────────────────────────────────────────────────────────────────
RETRIEVAL_TIMEOUT_SECONDS = 20.0
HYDE_TIMEOUT_SECONDS = 5.0
DEFAULT_TOP_K = 3
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

────────────────────────────────────────────────────────────────────────
KEY BEHAVIORS
────────────────────────────────────────────────────────────────────────
- retrieve() is the ONLY public async method called by the pipeline
- _generate_hypothetical_query() must never block — use asyncio properly
- _embed() is synchronous; wrap with asyncio.to_thread if called inside async context
- ChromaDB client must be initialized once in __init__, not per call
- If CHROMA_DB_PATH does not exist, create it (chromadb handles this automatically)
- self._kb_available check must happen BEFORE any retrieval attempt
- All ChromaDB queries must include where= filter for argument_type
  when _get_type_filter returns a non-empty list
- Model name: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
- Timeout on AsyncAnthropic() init only — NOT inside messages.create()
- Strip markdown fences before json.loads(): re.sub(r"```[a-z]*\n?|```", "", text).strip()

────────────────────────────────────────────────────────────────────────
ACCEPTANCE CRITERIA
────────────────────────────────────────────────────────────────────────
- retrieve() returns RetrievalContext within 2 seconds total
- Returns empty chunks (not an error) for CORRECT_PRONUNCIATION strategy
- Returns fallback context (fallback_used=True) when KB has 0 documents
- _generate_hypothetical_query() returns a coherent 2-sentence argument string
- Re-ranked results always sorted by composite score descending
- index_chunks() returns correct count of upserted documents
- No exception propagates from retrieve() — always returns RetrievalContext
- _get_fallback_context() returns real, useful stub content for all strategies
- ChromaDB collection persists across process restarts (persistent client)
""",
    },
    
}


# ── Revision loop utilities ───────────────────────────────────────────────────

def _extract_critical_issues(review_text: str) -> list[str]:
    """Extract all [CRITICAL] issue descriptions from a review."""
    return re.findall(r"\[CRITICAL\][^\n]+", review_text, re.IGNORECASE)


def _is_hard_stop(prev_review: str, curr_review: str, output_path: Path) -> tuple[bool, str]:
    """
    Check hard stop conditions. Returns (should_stop, reason).

    Hard stop conditions:
    1. Same CRITICAL issue appears in two consecutive reviews (agent is stuck)
    2. Output file fails py_compile after revision (broken Python)
    3. Output file is empty or missing after revision
    """
    # Condition 3: missing or empty output
    if not output_path.exists() or output_path.stat().st_size == 0:
        return True, "Hard stop: revision engineer produced empty or missing output file."

    # Condition 2: syntax error in revised file
    import py_compile, tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(output_path.read_text(encoding="utf-8"))
            tmp_path = tmp.name
        py_compile.compile(tmp_path, doraise=True)
        Path(tmp_path).unlink(missing_ok=True)
    except py_compile.PyCompileError as e:
        Path(tmp_path).unlink(missing_ok=True)
        return True, f"Hard stop: revised file has syntax error — {e}"

    # Condition 1: same CRITICAL issues as previous review
    if prev_review:
        prev_criticals = set(_extract_critical_issues(prev_review))
        curr_criticals = set(_extract_critical_issues(curr_review))
        repeated = prev_criticals & curr_criticals
        if repeated:
            issues_str = "; ".join(repeated)
            return True, f"Hard stop: same CRITICAL issue persists after revision — {issues_str}"

    return False, ""


def _extract_decision(review_text: str) -> str:
    if not review_text:
        return "REQUEST_CHANGES"
    match = re.search(r"DECISION:\s*(APPROVE|REQUEST_CHANGES)", review_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if "APPROVE" in review_text.upper() and "REQUEST_CHANGES" not in review_text.upper():
        return "APPROVE"
    return "REQUEST_CHANGES"


# ── Revision crew (built inline, not via @CrewBase) ───────────────────────────

def _build_revision_agents(agents_config_path: str):
    """Load agent configs from yaml and return revision_engineer agent."""
    import yaml
    with open(agents_config_path) as f:
        configs = yaml.safe_load(f)

    revision_cfg = configs.get("revision_engineer", {})
    return Agent(
        role=revision_cfg.get("role", "Revision Engineer"),
        goal=revision_cfg.get("goal", "Fix all issues identified by the code reviewer."),
        backstory=revision_cfg.get("backstory", ""),
        llm=revision_cfg.get("llm", "anthropic/claude-sonnet-4-5"),
        verbose=True,
    )


def _run_revision_cycle(module_def: dict, review_text: str, output_dir: Path, agents_config_path: str) -> tuple[str, str]:
    """
    Run one revision cycle: revision_engineer fixes issues, code_reviewer re-reviews.
    Returns (new_review_text, new_decision).
    """
    import yaml
    with open(agents_config_path) as f:
        configs = yaml.safe_load(f)

    revision_agent = _build_revision_agents(agents_config_path)

    module_name = module_def["module_name"]
    current_code_path = output_dir / module_name
    current_code = current_code_path.read_text(encoding="utf-8") if current_code_path.exists() else ""

    revision_task = Task(
        description=f"""
Fix all issues in the implementation of {module_name}.

REVIEW ISSUES TO FIX:
{review_text}

CURRENT IMPLEMENTATION:
{current_code}

Apply every fix listed in the review. Output the complete corrected module — full file, raw Python only.
No markdown, no backticks, no partial patches.
""",
        expected_output=f"Complete corrected Python module {module_name}. Raw Python only.",
        agent=revision_agent,
        output_file=str(output_dir / module_name),
    )

    revision_crew = Crew(
        agents=[revision_agent],
        tasks=[revision_task],
        process=Process.sequential,
        verbose=True,
    )
    revision_crew.kickoff()

    # Re-review the revised code
    reviewer_cfg = configs.get("code_reviewer", {})
    reviewer_agent = Agent(
        role=reviewer_cfg.get("role", "Code Reviewer"),
        goal=reviewer_cfg.get("goal", "Review implementation."),
        backstory=reviewer_cfg.get("backstory", ""),
        llm=reviewer_cfg.get("llm", "anthropic/claude-sonnet-4-5"),
        verbose=True,
    )

    revised_code = (output_dir / module_name).read_text(encoding="utf-8") if (output_dir / module_name).exists() else ""

    re_review_task = Task(
        description=f"""
Re-review the revised implementation of {module_name}.

Apply the same review criteria as before. Be especially strict about:
1. Completeness — every method must have a full body
2. Model name must use os.getenv
3. No timeout= inside messages.create()
4. TurnAnalysis attributes via analysis.argument.*
5. Markdown fences stripped before json.loads()

REVISED IMPLEMENTATION:
{revised_code}

Output format:
DECISION: APPROVE or DECISION: REQUEST_CHANGES

ISSUES (if any):
- [CRITICAL/MINOR] Description

SUMMARY: one paragraph.
""",
        expected_output="Structured review with DECISION: APPROVE or DECISION: REQUEST_CHANGES.",
        agent=reviewer_agent,
        output_file=str(output_dir / f"{module_name}_review.md"),
    )

    re_review_crew = Crew(
        agents=[reviewer_agent],
        tasks=[re_review_task],
        process=Process.sequential,
        verbose=True,
    )
    re_review_crew.kickoff()

    new_review = (output_dir / f"{module_name}_review.md").read_text(encoding="utf-8") \
        if (output_dir / f"{module_name}_review.md").exists() else ""
    new_decision = _extract_decision(new_review)
    return new_review, new_decision


# ── Runner ────────────────────────────────────────────────────────────────────

def run(module_key: str = "turn_analyzer", use_github: bool = True):
    if module_key not in MODULES:
        print(f"Unknown module: {module_key}. Available: {list(MODULES.keys())}")
        sys.exit(1)

    module_def = MODULES[module_key]
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    agents_config_path = Path(__file__).parent / "config" / "agents.yaml"

    print(f"\n{'='*60}")
    print(f"  SpeakFlow Dev Team — Building: {module_def['module_name']}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Run the initial crew (design → code → review → test → frontend) ──────
    result = SpeakFlowDevTeam().crew().kickoff(inputs=module_def)

    # ── Read initial review ───────────────────────────────────────────────────
    review_path = output_dir / f"{module_def['module_name']}_review.md"
    review_text = review_path.read_text(encoding="utf-8") if review_path.exists() else ""
    decision = _extract_decision(review_text)

    print(f"\n[RevisionLoop] Initial reviewer decision: {decision}")

    # ── Revision loop ─────────────────────────────────────────────────────────
    revision_count = 0
    prev_review = ""

    while decision == "REQUEST_CHANGES" and revision_count < MAX_REVISIONS:
        revision_count += 1
        print(f"\n[RevisionLoop] Starting revision {revision_count}/{MAX_REVISIONS}...")

        new_review, new_decision = _run_revision_cycle(
            module_def, review_text, output_dir, str(agents_config_path)
        )

        # Check hard stop conditions
        output_path = output_dir / module_def["module_name"]
        should_stop, stop_reason = _is_hard_stop(prev_review, new_review, output_path)

        if should_stop:
            print(f"\n[RevisionLoop] ⛔ {stop_reason}")
            print("[RevisionLoop] Aborting pipeline. Files saved to output/ for manual inspection.")
            print(f"[RevisionLoop] Last review:\n{new_review[:500]}...")
            return None

        prev_review = review_text
        review_text = new_review
        decision = new_decision

        print(f"[RevisionLoop] Revision {revision_count} reviewer decision: {decision}")

    if decision == "REQUEST_CHANGES":
        print(f"\n[RevisionLoop] ⚠️  Reached MAX_REVISIONS ({MAX_REVISIONS}) without APPROVE.")
        print("[RevisionLoop] Opening PR anyway for human review.")
    else:
        print(f"\n[RevisionLoop] ✅ APPROVE received after {revision_count} revision(s).")

    print(f"\n{'='*60}")
    print(f"  Dev Team completed: {module_def['module_name']}")
    print(f"  Final decision: {decision}  |  Revisions: {revision_count}")
    print(f"  Output files in: {output_dir.absolute()}")
    print(f"{'='*60}\n")

    # ── GitHub PR integration ─────────────────────────────────────────────────
    if use_github and os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO"):
        pr_manager = GitHubPRManager(
            token=os.getenv("GITHUB_TOKEN"),
            repo=os.getenv("GITHUB_REPO"),
        )
        pr = pr_manager.create_pr_from_output(
            module_name=module_def["module_name"],
            output_dir=output_dir,
            review_text=review_text,
        )
        if pr is None:
            print("[GitHub] PR creation aborted (syntax check failed). Fix output files manually.")
    else:
        if use_github:
            print("  [GitHub] Skipped — set GITHUB_TOKEN and GITHUB_REPO to enable PR creation.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeakFlow Dev Team Runner")
    parser.add_argument("--module", default="turn_analyzer", help="Module key to build")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub PR creation")
    args = parser.parse_args()

    run(module_key=args.module, use_github=not args.skip_github)