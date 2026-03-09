"""
shared_types.py
===============
Single source of truth for all data models shared across SpeakFlow AI modules.

Import pattern (in each module):

    turn_analyzer.py:
        from shared_types import (
            TurnInput, ArgumentResult, ErrorSeverity, WordError,
            PronunciationResult, TurnAnalysis, TurnIntent,
        )

    coach_policy.py:
        from shared_types import (
            TurnAnalysis, CoachingStrategy, CoachingAction, SessionContext,
        )

    response_generator.py:
        from shared_types import (
            CoachingStrategy, CoachingAction, ToneMode,
            ResponseRequest, GeneratedResponse,
        )

    session_evaluator.py:
        from shared_types import (
            SessionTurn, SessionMetadata, CoachingStrategy,
            ArgumentResult, PronunciationResult,
        )

Module ownership:
    turn_analyzer.py      → produces  TurnAnalysis
    coach_policy.py       → consumes  TurnAnalysis,    produces CoachingAction
    response_generator.py → consumes  CoachingAction,  produces GeneratedResponse
    session_evaluator.py  → consumes  List[SessionTurn] (bundles all three above)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ── 1. TurnAnalyzer output types ──────────────────────────────────────────────

@dataclass
class TurnInput:
    transcript: str
    session_id: str
    turn_number: int
    topic: str
    user_position: str          # "for" or "against"
    audio_path: str
    prior_turns: List[str]


@dataclass
class ArgumentResult:
    # ── Four-dimension scores (each 0.0–1.0) ──────────────────────────────────
    clarity_score:   float      # 30% weight — clear position, complete sentences
    reasoning_score: float      # 30% weight — logical reasons supporting the claim
    depth_score:     float      # 10% weight — examples, analogies, elaboration
    fluency_score_arg: float    # 30% weight — grammar, coherence, no major errors

    # ── Weighted total ────────────────────────────────────────────────────────
    argument_score: float       # 0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency

    # ── Diagnostic fields ─────────────────────────────────────────────────────
    has_claim:     bool
    has_reasoning: bool
    has_evidence:  bool         # kept for backward compat, maps to depth_score > 0.3

    logical_gaps:      List[str]
    vocabulary_flags:  List[str]
    summary:           str      # one sentence, encouraging tone

    # ── Per-dimension feedback (shown to student) ─────────────────────────────
    clarity_feedback:   str
    reasoning_feedback: str
    depth_feedback:     str
    fluency_feedback:   str


class TurnIntent(str, Enum):
    """
    Intent of the student's input this turn.
    Detected before scoring to route to the correct processing path.
    """
    DEBATE_STATEMENT  = "debate_statement"   # Normal argument turn → score + coach question
    META_QUESTION     = "meta_question"      # Student asking coach something → answer + re-prompt
    OFF_TOPIC         = "off_topic"          # Irrelevant input → redirect


class ErrorSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class WordError:
    word: str
    expected_ipa: str
    actual_ipa: str
    severity: ErrorSeverity


@dataclass
class PronunciationResult:
    mispronounced_words: List[WordError]
    fluency_score: float        # 0.0 – 1.0
    target_phonemes: List[str]  # e.g. ["/r/", "/θ/"]


@dataclass
class TurnAnalysis:
    turn_input: TurnInput
    argument: ArgumentResult
    pronunciation: PronunciationResult
    timestamp: datetime
    latency_ms: int


# ── 2. CoachPolicyAgent output types ─────────────────────────────────────────

class CoachingStrategy(str, Enum):
    """
    Five strategies the coach can apply on any given turn.
    Chosen by CoachPolicyAgent, consumed by ResponseGenerator.
    """
    PROBE                 = "PROBE"                  # Ask a deepening question
    CHALLENGE             = "CHALLENGE"              # Present a counter-argument
    REDIRECT              = "REDIRECT"               # Bring student back on topic
    PRAISE                = "PRAISE"                 # Reinforce a strong argument
    CORRECT_PRONUNCIATION = "CORRECT_PRONUNCIATION"  # Model correct pronunciation


@dataclass
class CoachingAction:
    """
    Pure decision record — NO natural language response included.
    ResponseGenerator reads this and generates the actual text.
    """
    strategy: CoachingStrategy
    intent: str                         # Human-readable reason for the strategy choice
    target_claim: Optional[str]         # The specific claim to probe or challenge (if relevant)
    target_word: Optional[str]          # Mispronounced word to correct (CORRECT_PRONUNCIATION only)
    target_phoneme: Optional[str]       # Phoneme to model   (CORRECT_PRONUNCIATION only)
    argument_score: float               # Passed through from TurnAnalysis for downstream use
    pronunciation_score: float          # Fluency score passed through from TurnAnalysis
    difficulty_delta: int               # -1 / 0 / +1 — difficulty adjustment signal
    turn_number: int
    topic: str
    user_position: str                  # "for" or "against"
    prior_coach_responses: List[str]    # Last 3 coach response texts, for continuity


# ── 3. CoachPolicyAgent session context ──────────────────────────────────────

@dataclass
class SessionContext:
    """
    Running session state passed into CoachPolicyAgent.decide() on every turn.
    Tracks history needed for strategy selection and difficulty adjustment.
    """
    session_id: str
    topic: str
    user_position: str                  # "for" or "against"
    turn_number: int
    coaching_history: List[CoachingStrategy]
    argument_scores: List[float]
    last_coach_question: str = ""       # prevents coach repeating same question
    last_turn_intent: str = ""          # TurnIntent value of previous turn


# ── 4. ResponseGenerator input/output types ───────────────────────────────────

class ToneMode(str, Enum):
    """Tone applied by ResponseGenerator when generating natural language."""
    SOCRATIC   = "socratic"     # Questioning approach
    CHALLENGING = "challenging" # Pushback approach
    AFFIRMING  = "affirming"    # Praise approach


@dataclass
class ResponseRequest:
    """Input to ResponseGenerator.generate_response()."""
    coaching_action: CoachingAction
    topic: str
    user_position: str
    prior_responses: List[str]          # Last 3 coach response texts (repetition avoidance)
    turn_number: int


@dataclass
class GeneratedResponse:
    """Natural language reply produced by ResponseGenerator."""
    text: str
    strategy_used: CoachingStrategy
    tone: ToneMode
    follow_up_prompt: Optional[str]     # Optional trailing sentence to invite continuation
    estimated_speaking_seconds: float
    timestamp: datetime
    latency_ms: int


# ── 5. Session-level bundle (consumed by SessionEvaluator) ───────────────────

@dataclass
class SessionTurn:
    """
    Complete record of one debate turn — everything the SessionEvaluator needs.
    Assembled by the LangGraph pipeline after all three modules have run.
    """
    turn_number: int
    analysis: TurnAnalysis
    coaching_action: CoachingAction
    response: GeneratedResponse


@dataclass
class SessionMetadata:
    session_id: str
    topic: str
    user_position: str          # "for" or "against"
    started_at: datetime
    ended_at: Optional[datetime]
    total_turns: int