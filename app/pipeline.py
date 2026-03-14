"""
pipeline.py — SpeakFlow AI LangGraph Orchestration Layer
=========================================================
Replaces the manual asyncio.gather chains in app.py with a typed StateGraph.

Graph topology (per turn):
    START
      └─ intent_node          (no API — keyword routing)
           ├─ META_QUESTION  → meta_handler_node → END
           ├─ OFF_TOPIC      → off_topic_node    → END
           └─ DEBATE_STATEMENT
                └─ fan-out via Send API (true parallelism):
                     ├─ score_node          (Claude: 4 scores + bool flags)
                     ├─ summary_node        (Claude: feedback text + summary)
                     └─ pronunciation_node  (MFA stub → fluency score)
                └─ merge_analysis_node  (assemble TurnAnalysis from 3 partial results)
                └─ fan-out (parallel):
                     ├─ coach_policy_node  (rule-based strategy selection)
                     └─ rag_node           (HyDE + ChromaDB retrieval)
                └─ response_node        (Claude: coach text + improved + tips)
                └─ update_session_node  (append scores/history to state)
                └─ END

Key design decisions:
  - score_node and summary_node replace the single _analyze_argument() Claude call,
    enabling CoachPolicyAgent to start as soon as scores are ready (no waiting for text).
  - All session state (prior_turns, coaching_history, argument_scores) lives in
    SpeakFlowState, making the pipeline stateless and resumable via MemorySaver.
  - TurnAnalyzer, CoachPolicyAgent, ResponseGenerator, RAGRetriever instances are
    created once and injected via pipeline config — no re-initialisation per turn.
  - app.py calls pipeline.invoke(state) and reads output fields directly.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from shared_types import (
    TurnInput,
    TurnIntent,
    TurnAnalysis,
    ArgumentResult,
    PronunciationResult,
    CoachingAction,
    CoachingStrategy,
    SessionContext,
    ResponseRequest,
    GeneratedResponse,
)

logger = logging.getLogger(__name__)

# ── State reducer helpers ─────────────────────────────────────────────────────

def _keep_last(a: Any, b: Any) -> Any:
    """Reducer: always take the newer value (used for most fields)."""
    return b if b is not None else a


def _append(a: list, b: list) -> list:
    """Reducer: extend list (used for history fields)."""
    if a is None:
        return b or []
    if b is None:
        return a
    return a + b


# ── SpeakFlowState ────────────────────────────────────────────────────────────

class SpeakFlowState(TypedDict, total=False):
    # ── Input (set by caller before invoke) ───────────────────────────────────
    transcript:        str
    topic:             str
    user_position:     str           # "for" or "against"
    audio_path:        str
    session_id:        str

    # ── Persisted across turns (managed by MemorySaver) ───────────────────────
    turn_number:       Annotated[int,              _keep_last]
    prior_turns:       Annotated[list[dict],       _keep_last]
    coaching_history:  Annotated[list[str],        _keep_last]   # CoachingStrategy.value
    argument_scores:   Annotated[list[float],      _keep_last]
    prior_responses:   Annotated[list[str],        _keep_last]
    last_coach_question: Annotated[str,            _keep_last]
    last_turn_intent:  Annotated[str,              _keep_last]
    session_transcripts: Annotated[list[str],      _keep_last]
    session_language_tips: Annotated[list[str],    _keep_last]

    # ── Partial analysis results (written by fan-out nodes) ───────────────────
    # These hold intermediate data before merge_analysis_node assembles TurnAnalysis.
    _score_result:     Annotated[Optional[dict],   _keep_last]   # raw score dict from score_node
    _summary_result:   Annotated[Optional[dict],   _keep_last]   # raw feedback dict from summary_node
    _pronunciation_result: Annotated[Optional[PronunciationResult], _keep_last]

    # ── Pipeline outputs (written by nodes, read by app.py) ───────────────────
    turn_intent:       Annotated[str,              _keep_last]
    turn_analysis:     Annotated[Optional[TurnAnalysis],    _keep_last]
    coaching_action:   Annotated[Optional[CoachingAction],  _keep_last]
    retrieval_context: Annotated[Any,              _keep_last]   # RetrievalContext | None
    coach_text:        Annotated[str,              _keep_last]
    improved_text:     Annotated[str,              _keep_last]
    language_tips:     Annotated[str,              _keep_last]
    pronunciation_text: Annotated[str,             _keep_last]
    status_message:    Annotated[str,              _keep_last]
    error:             Annotated[Optional[str],    _keep_last]


# ── Pipeline class ────────────────────────────────────────────────────────────

class SpeakFlowPipeline:
    """
    LangGraph-based orchestration layer for SpeakFlow AI.

    Usage:
        pipeline = SpeakFlowPipeline()
        result = await pipeline.ainvoke({
            "transcript": "...",
            "topic": "...",
            "user_position": "for",
            "session_id": "abc123",
            # persisted state fields are loaded automatically by MemorySaver
        }, config={"configurable": {"thread_id": session_id}})
        coach_text = result["coach_text"]
    """

    def __init__(self):
        from turn_analyzer import TurnAnalyzer
        from coach_policy import CoachPolicyAgent
        from response_generator import ResponseGenerator
        from rag_retriever import RAGRetriever
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._turn_analyzer  = TurnAnalyzer(anthropic_api_key=api_key)
        self._coach          = CoachPolicyAgent(mfa_enabled=False)
        self._response_gen   = ResponseGenerator()
        self._rag_retriever  = RAGRetriever()

        # Dedicated client for all pipeline API calls.
        # max_retries=1: handles transient connection errors (observed as
        # "Connection error" on Turn 2/3) without the 20s+ blowout from
        # the default max_retries=2 exponential backoff.
        self._pipeline_client = AsyncAnthropic(
            api_key=api_key,
            max_retries=1,
            timeout=float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30")),
        )
        self._pipeline_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        # Unify: replace each module's internal Anthropic client with
        # _pipeline_client so ALL API calls in the pipeline share the same
        # retry policy. Without this, _gen_response (via ResponseGenerator)
        # and rag_node HyDE (via RAGRetriever) still use max_retries=2,
        # which caused the 22s+ Turn 1 blowout.
        self._response_gen._client   = self._pipeline_client
        self._rag_retriever._anthropic = self._pipeline_client

        self._graph = self._build_graph()

    # ── Graph builder ─────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(SpeakFlowState)

        # Register nodes
        g.add_node("intent_node",         self._intent_node)
        g.add_node("meta_handler_node",   self._meta_handler_node)
        g.add_node("off_topic_node",      self._off_topic_node)
        g.add_node("score_node",          self._score_node)
        g.add_node("summary_node",        self._summary_node)
        g.add_node("pronunciation_node",  self._pronunciation_node)
        g.add_node("merge_analysis_node", self._merge_analysis_node)
        g.add_node("coach_policy_node",   self._coach_policy_node)
        g.add_node("rag_node",            self._rag_node)
        g.add_node("response_node",       self._response_node)
        g.add_node("update_session_node", self._update_session_node)

        # Edges
        g.add_edge(START, "intent_node")
        g.add_conditional_edges("intent_node", self._route_by_intent)
        g.add_edge("meta_handler_node",   END)
        g.add_edge("off_topic_node",      END)

        # fan-out: intent_node → [score_node, summary_node, pronunciation_node]
        # handled inside _route_by_intent via Send

        # fan-in: all three analysis nodes → merge_analysis_node
        g.add_edge("score_node",         "merge_analysis_node")
        g.add_edge("summary_node",       "merge_analysis_node")
        g.add_edge("pronunciation_node", "merge_analysis_node")

        # parallel: merge → [coach_policy_node, rag_node] → response_node
        g.add_edge("merge_analysis_node",  "coach_policy_node")
        g.add_edge("merge_analysis_node",  "rag_node")
        g.add_edge("coach_policy_node",    "response_node")
        g.add_edge("rag_node",             "response_node")

        g.add_edge("response_node",        "update_session_node")
        g.add_edge("update_session_node",  END)

        # Register custom dataclasses/enums with MemorySaver to suppress
        # "Deserializing unregistered type" warnings (non-fatal, but noisy).
        import importlib
        _allowed: list[tuple[str, str]] = []
        for mod_name in ("shared_types", "rag_retriever"):
            try:
                m = importlib.import_module(mod_name)
                for attr in dir(m):
                    obj = getattr(m, attr)
                    if isinstance(obj, type):
                        _allowed.append((mod_name, attr))
            except Exception:
                pass

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        serde = JsonPlusSerializer().with_msgpack_allowlist(_allowed) if _allowed else JsonPlusSerializer()
        memory = MemorySaver(serde=serde)

        return g.compile(checkpointer=memory)

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_by_intent(self, state: SpeakFlowState):
        """
        Conditional edge after intent_node.
        - META_QUESTION / OFF_TOPIC → single terminal node
        - DEBATE_STATEMENT → fan-out to 3 parallel analysis nodes via Send
        """
        intent = state.get("turn_intent", TurnIntent.DEBATE_STATEMENT)

        if intent == TurnIntent.META_QUESTION:
            return "meta_handler_node"
        if intent == TurnIntent.OFF_TOPIC:
            return "off_topic_node"

        # Fan-out: send same state slice to all three analysis nodes
        return [
            Send("score_node",         state),
            Send("summary_node",       state),
            Send("pronunciation_node", state),
        ]

    # ── Node: intent ──────────────────────────────────────────────────────────

    async def _intent_node(self, state: SpeakFlowState) -> dict:
        """Detect intent from transcript — no API call.

        turn_number is only incremented for DEBATE_STATEMENT turns.
        META_QUESTION and OFF_TOPIC do not count as debate turns.
        """
        transcript = state.get("transcript", "")
        intent = self._turn_analyzer._detect_intent(transcript)
        current = state.get("turn_number") or 0
        turn_number = current + 1 if intent == TurnIntent.DEBATE_STATEMENT else current
        logger.info(f"[intent_node] turn={turn_number} intent={intent}")
        return {
            "turn_intent": intent,
            "turn_number": turn_number,
        }

    # ── Node: meta handler ────────────────────────────────────────────────────

    async def _meta_handler_node(self, state: SpeakFlowState) -> dict:
        """Student pushed back or asked a clarifying question.
        Generates a direct answer that acknowledges the student and redirects
        them back to their debate argument — always within the correct topic/position.
        Uses pipeline_client directly (bypasses ResponseGenerator's retry policy).
        """
        transcript = state.get("transcript", "")
        topic      = state.get("topic", "")
        position   = state.get("user_position", "for")
        last_q     = state.get("last_coach_question", "")
        prior_resp = state.get("prior_responses") or []

        # Explicit debate context in prompt — prevents coach from drifting
        # to wrong side of the argument (observed: coach suggested "benefits of
        # social media" when student position was FOR "harms").
        prompt = (
            f'You are a debate coach. The debate topic is: "{topic}"\n'
            f'The student is arguing the {position.upper()} side '
            f'(i.e. they believe the statement is TRUE).\n\n'
            f'You previously asked: "{last_q}"\n'
            f'The student pushed back: "{transcript}"\n\n'
            f'Rules:\n'
            f'- Acknowledge the student briefly (1 sentence)\n'
            f'- Do NOT repeat or rephrase your previous question\n'
            f'- Give them ONE concrete talking point to develop, '
            f'  consistent with the {position.upper()} position on "{topic}"\n'
            f'- Under 60 words total\n'
            f'- Do not mention "policy" unless the topic itself mentions policy'
        )

        coach_text = "[Coach]\n\nYou're right — let's move on. Try giving a concrete example of the harm."
        new_last_q = coach_text

        try:
            response = await self._pipeline_client.messages.create(
                model=self._pipeline_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = response.content[0].text.strip()
            coach_text = f"[Coach]\n\n{reply}"
            new_last_q = reply
        except Exception as e:
            logger.error(f"[meta_handler_node] ({type(e).__name__}): {e}")

        return {
            "coach_text":          coach_text,
            "improved_text":       "",
            "language_tips":       "",
            "pronunciation_text":  "",
            "last_coach_question": new_last_q,
            "status_message":      f"💬 Turn {state.get('turn_number', 1)} — Coach answered your question.",
        }

    # ── Node: off topic ───────────────────────────────────────────────────────

    async def _off_topic_node(self, state: SpeakFlowState) -> dict:
        topic = state.get("topic", "the debate topic")
        return {
            "coach_text":         f"[Coach]\n\nLet's focus on the debate. Try sharing your view on: {topic}",
            "improved_text":      "",
            "language_tips":      "",
            "pronunciation_text": "",
            "status_message":     "⚠️ Please provide a longer argument.",
        }

    # ── Node: score (fast scores only) ───────────────────────────────────────

    async def _score_node(self, state: SpeakFlowState) -> dict:
        """
        First half of the TurnAnalyzer split.
        Calls Claude for numeric scores + boolean flags ONLY.
        OPT-1: uses self._pipeline_client (max_retries=0) — no retry jitter.
        OPT-2: compressed prompt (~120 tokens vs ~400) — faster TTFT.
        """
        transcript = state.get("transcript", "")
        topic      = state.get("topic", "")
        position   = state.get("user_position", "for")
        turn_num   = state.get("turn_number", 1)
        prior      = state.get("prior_turns") or []

        prior_ctx = ""
        if prior:
            last = prior[-1]
            s = last.get("summary", "") if isinstance(last, dict) else str(last)
            if s:
                prior_ctx = f"\nPrev: {s[:120]}"

        # OPT-2: Compressed prompt. Full rubric lives in summary_node which
        # runs in parallel — score_node only needs numbers and booleans fast.
        prompt = (
            f'Topic: {topic} | Position: {position} | Turn: {turn_num}{prior_ctx}\n'
            f'Student: "{transcript[:400]}"\n\n'
            'Score generously (L2 learner). Each 0.0-1.0.\n'
            'clarity=clear position  reasoning=logical reasons  '
            'depth=examples  fluency=grammar\n'
            'argument_score=0.3*clarity+0.3*reasoning+0.1*depth+0.3*fluency\n'
            'has_claim=stated position  has_reasoning=gave reason  '
            'has_evidence=depth>=0.4\n'
            'logical_gaps=up to 2 short phrases  vocabulary_flags=up to 3 weak words\n\n'
            'JSON only:\n'
            '{"clarity_score":0.0,"reasoning_score":0.0,"depth_score":0.0,'
            '"fluency_score_arg":0.0,"argument_score":0.0,'
            '"has_claim":false,"has_reasoning":false,"has_evidence":false,'
            '"logical_gaps":[],"vocabulary_flags":[]}'
        )

        try:
            import json, re as _re
            response = await self._pipeline_client.messages.create(
                model=self._pipeline_model,
                max_tokens=200,
                system="JSON only. No markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            raw = _re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
            logger.info(f"[score_node] turn={turn_num} score={data.get('argument_score', 0):.2f}")
            return {"_score_result": data}
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[score_node] failed ({error_type}): {e}")
            return {"_score_result": {
                "clarity_score": 0.0, "reasoning_score": 0.0,
                "depth_score": 0.0, "fluency_score_arg": 0.0,
                "argument_score": 0.0, "has_claim": False,
                "has_reasoning": False, "has_evidence": False,
                "logical_gaps": [], "vocabulary_flags": [],
            }}

    # ── Node: summary (feedback text only) ───────────────────────────────────

    async def _summary_node(self, state: SpeakFlowState) -> dict:
        """
        Second half of the TurnAnalyzer split.
        Calls Claude for per-dimension feedback text + summary sentence ONLY.
        Runs in parallel with score_node — neither blocks the other.
        OPT-1: uses self._pipeline_client (max_retries=0).
        """
        transcript = state.get("transcript", "")
        topic      = state.get("topic", "")
        position   = state.get("user_position", "for")
        turn_num   = state.get("turn_number", 1)

        prompt = (
            f'Topic: {topic} | Position: {position}\n'
            f'Student said: "{transcript[:400]}"\n\n'
            'Write short coaching feedback. Each value: one encouraging phrase, max 10 words.\n'
            'JSON only:\n'
            '{"clarity_feedback":"...","reasoning_feedback":"...",'
            '"depth_feedback":"...","fluency_feedback":"...",'
            '"summary":"one encouraging sentence max 20 words"}'
        )

        try:
            import json, re as _re
            response = await self._pipeline_client.messages.create(
                model=self._pipeline_model,
                max_tokens=200,
                system="JSON only. No markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            raw = _re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
            logger.info(f"[summary_node] turn={turn_num} done")
            return {"_summary_result": data}
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[summary_node] failed ({error_type}): {e}")
            return {"_summary_result": {
                "clarity_feedback":   "Keep your position clear.",
                "reasoning_feedback": "Try to add a reason.",
                "depth_feedback":     "Add an example next time.",
                "fluency_feedback":   "Good effort with your English.",
                "summary":            "Keep going — you're building your argument well.",
            }}

    # ── Node: pronunciation ───────────────────────────────────────────────────

    async def _pronunciation_node(self, state: SpeakFlowState) -> dict:
        """Run MFA stub (or real MFA) — runs in parallel with score/summary nodes."""
        turn_input = TurnInput(
            transcript  = state.get("transcript", ""),
            session_id  = state.get("session_id", ""),
            turn_number = state.get("turn_number", 1),
            topic       = state.get("topic", ""),
            user_position = state.get("user_position", "for"),
            audio_path  = state.get("audio_path", ""),
            prior_turns = state.get("prior_turns") or [],
        )
        try:
            result = await self._turn_analyzer._analyze_pronunciation(turn_input)
        except Exception as e:
            logger.error(f"[pronunciation_node] {e}")
            result = self._turn_analyzer._create_default_pronunciation_result()
        return {"_pronunciation_result": result}

    # ── Node: merge analysis ──────────────────────────────────────────────────

    async def _merge_analysis_node(self, state: SpeakFlowState) -> dict:
        """
        Fan-in: assemble TurnAnalysis from the three parallel partial results.
        Waits for all three nodes to complete (LangGraph guarantees this before
        calling merge when all incoming edges are satisfied).
        """
        score_data   = state.get("_score_result") or {}
        summary_data = state.get("_summary_result") or {}
        pron_result  = state.get("_pronunciation_result")

        if pron_result is None:
            pron_result = self._turn_analyzer._create_default_pronunciation_result()

        clarity   = float(score_data.get("clarity_score",    0.0))
        reasoning = float(score_data.get("reasoning_score",  0.0))
        depth     = float(score_data.get("depth_score",      0.0))
        fluency   = float(score_data.get("fluency_score_arg",0.0))
        arg_score = float(score_data.get("argument_score",   round(0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency, 3)))

        argument = ArgumentResult(
            clarity_score     = clarity,
            reasoning_score   = reasoning,
            depth_score       = depth,
            fluency_score_arg = fluency,
            argument_score    = arg_score,
            has_claim         = bool(score_data.get("has_claim",    False)),
            has_reasoning     = bool(score_data.get("has_reasoning", False)),
            has_evidence      = bool(score_data.get("has_evidence",  depth >= 0.4)),
            logical_gaps      = score_data.get("logical_gaps",    []),
            vocabulary_flags  = score_data.get("vocabulary_flags", []),
            clarity_feedback  = summary_data.get("clarity_feedback",   ""),
            reasoning_feedback= summary_data.get("reasoning_feedback", ""),
            depth_feedback    = summary_data.get("depth_feedback",     ""),
            fluency_feedback  = summary_data.get("fluency_feedback",   ""),
            summary           = summary_data.get("summary",            ""),
        )

        turn_input = TurnInput(
            transcript    = state.get("transcript", ""),
            session_id    = state.get("session_id", ""),
            turn_number   = state.get("turn_number", 1),
            topic         = state.get("topic", ""),
            user_position = state.get("user_position", "for"),
            audio_path    = state.get("audio_path", ""),
            prior_turns   = state.get("prior_turns") or [],
        )

        analysis = TurnAnalysis(
            turn_input    = turn_input,
            argument      = argument,
            pronunciation = pron_result,
            timestamp     = datetime.now(),
            latency_ms    = 0,
        )

        logger.info(f"[merge_analysis_node] turn={state.get('turn_number')} "
                    f"score={arg_score:.2f} claim={argument.has_claim}")
        return {"turn_analysis": analysis}

    # ── Node: coach policy ────────────────────────────────────────────────────

    async def _coach_policy_node(self, state: SpeakFlowState) -> dict:
        """Select coaching strategy — rule-based, no API call."""
        analysis = state.get("turn_analysis")
        if analysis is None:
            logger.warning("[coach_policy_node] no turn_analysis in state")
            return {"coaching_action": None}

        coaching_history_vals = state.get("coaching_history") or []
        coaching_history = [
            CoachingStrategy(v) for v in coaching_history_vals
            if v in CoachingStrategy._value2member_map_
        ]

        session_ctx = SessionContext(
            session_id        = state.get("session_id", ""),
            topic             = state.get("topic", ""),
            user_position     = state.get("user_position", "for"),
            turn_number       = state.get("turn_number", 1),
            coaching_history  = coaching_history,
            argument_scores   = list(state.get("argument_scores") or []),
            last_coach_question = state.get("last_coach_question", ""),
            last_turn_intent  = state.get("last_turn_intent", ""),
        )

        try:
            action = await self._coach.decide(analysis, session_ctx)
        except Exception as e:
            logger.error(f"[coach_policy_node] {e}")
            action = self._coach._create_default_action(session_ctx)

        # Build intent string (mirrors app.py logic)
        arg = analysis.argument
        dim_lines = [
            f"  {lbl}: {val:.1f} {'✓' if val >= thr else '✗'}"
            for lbl, val, thr in [
                ("Clarity",   arg.clarity_score,    0.5),
                ("Reasoning", arg.reasoning_score,  0.5),
                ("Depth",     arg.depth_score,       0.4),
                ("Fluency",   arg.fluency_score_arg, 0.5),
            ]
        ]
        weakest = [
            lbl for lbl, weak in [
                ("clearer position statement",        arg.clarity_score < 0.5),
                ("logical reasoning explaining WHY",  arg.reasoning_score < 0.5),
                ("a concrete example or elaboration", arg.depth_score < 0.4),
                ("cleaner grammar and connectives",   arg.fluency_score_arg < 0.5),
            ] if weak
        ]
        action.intent = "\n".join([
            f"Topic: {state.get('topic', '')}",
            f"Student position: {state.get('user_position', 'for')}",
            f"Turn {state.get('turn_number', 1)}. Student just said:",
            f'  "{(state.get("transcript") or "")[:300]}"',
            "Scores:", *dim_lines,
            f"Coach summary: {arg.summary}",
            f"Weakest areas: {', '.join(weakest) if weakest else 'none — strong'}",
            f"Coach last asked: \"{state.get('last_coach_question', '')}\"",
            "Do NOT repeat or rephrase the coach's last question.",
        ])

        logger.info(f"[coach_policy_node] strategy={action.strategy.value}")
        return {"coaching_action": action}

    # ── Node: RAG ─────────────────────────────────────────────────────────────

    async def _rag_node(self, state: SpeakFlowState) -> dict:
        """HyDE retrieval — runs in parallel with coach_policy_node."""
        analysis = state.get("turn_analysis")
        if analysis is None:
            return {"retrieval_context": None}

        # Build a minimal pre-action for RAG (strategy resolved after merge,
        # so we use PROBE as default — good enough for HyDE query generation)
        pre_action = CoachingAction(
            strategy          = CoachingStrategy.PROBE,
            topic             = state.get("topic", ""),
            user_position     = state.get("user_position", "for"),
            intent            = "",
            target_claim=None, target_word=None, target_phoneme=None,
            argument_score    = analysis.argument.argument_score,
            pronunciation_score = 1.0,
            difficulty_delta  = 0,
            turn_number       = state.get("turn_number", 1),
            prior_coach_responses = list(state.get("prior_responses") or [])[-3:],
        )

        try:
            ctx = await self._rag_retriever.retrieve(pre_action, analysis)
            logger.info(f"[rag_node] fallback={ctx.fallback_used} chunks={len(ctx.chunks)}")
        except Exception as e:
            logger.error(f"[rag_node] {e}")
            ctx = None
        return {"retrieval_context": ctx}

    # ── Node: response ────────────────────────────────────────────────────────

    # Hard ceiling for the entire response_node. generate_improved_version and
    # generate_language_tips previously had no outer timeout — a single API
    # retry inside ResponseGenerator could add 20+ seconds (observed in Turn 1
    # log: 16:14:25 → 16:14:47). 12s covers normal latency (3-4s per call,
    # 3 calls in parallel) while hard-stopping runaway retries.
    RESPONSE_NODE_TIMEOUT = 12.0

    async def _response_node(self, state: SpeakFlowState) -> dict:
        """Generate coach text, improved version, and language tips in parallel.
        Total budget: RESPONSE_NODE_TIMEOUT seconds. Each sub-call uses
        self._pipeline_client (max_retries=1) so a single connection error
        gets one retry without the 20s+ blowout from the default retry policy.
        """
        action  = state.get("coaching_action")
        analysis = state.get("turn_analysis")

        if action is None or analysis is None:
            return {
                "coach_text":        "",
                "improved_text":     "",
                "language_tips":     "",
                "pronunciation_text": "",
                "status_message":    "⚠️ Analysis unavailable.",
            }

        action.prior_coach_responses = list(state.get("prior_responses") or [])[-3:]

        request = ResponseRequest(
            coaching_action   = action,
            topic             = state.get("topic", ""),
            user_position     = state.get("user_position", "for"),
            prior_responses   = list(state.get("prior_responses") or []),
            turn_number       = state.get("turn_number", 1),
            retrieval_context = state.get("retrieval_context"),
        )

        arg = analysis.argument
        transcript = state.get("transcript", "")
        topic      = state.get("topic", "")
        position   = state.get("user_position", "for")

        async def _gen_response():
            try:
                resp = await self._response_gen.generate_response(request)
                return resp
            except Exception as e:
                logger.error(f"[response_node] generate_response ({type(e).__name__}): {e}")
                return None

        async def _gen_improved():
            """Direct pipeline_client call — bypasses ResponseGenerator's
            internal retry policy to stay within RESPONSE_NODE_TIMEOUT."""
            try:
                prompt = (
                    f'Rewrite this student debate argument in cleaner English. '
                    f'Keep the same position and ideas. Max 3 sentences.\n\n'
                    f'Topic: {topic} | Position: {position}\n'
                    f'Original: "{transcript[:400]}"\n\n'
                    f'Vocabulary to improve: {arg.vocabulary_flags or "none"}\n\n'
                    f'Output the improved version only, no explanation.'
                )
                r = await self._pipeline_client.messages.create(
                    model=self._pipeline_model, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text.strip()
            except Exception as e:
                logger.error(f"[response_node] improved_version ({type(e).__name__}): {e}")
                return ""

        async def _gen_tips():
            """Direct pipeline_client call — bypasses ResponseGenerator's
            internal retry policy to stay within RESPONSE_NODE_TIMEOUT."""
            try:
                feedback_ctx = " | ".join(filter(None, [
                    arg.clarity_feedback, arg.fluency_feedback
                ]))
                prompt = (
                    f'Give ONE specific English language tip for this debate student. '
                    f'Max 2 sentences. Focus on grammar or phrasing, not argument content.\n\n'
                    f'Student said: "{transcript[:300]}"\n'
                    f'Feedback context: {feedback_ctx or "general improvement"}\n\n'
                    f'Output the tip only.'
                )
                r = await self._pipeline_client.messages.create(
                    model=self._pipeline_model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text.strip()
            except Exception as e:
                logger.error(f"[response_node] language_tips ({type(e).__name__}): {e}")
                return ""

        try:
            resp_obj, improved, tips = await asyncio.wait_for(
                asyncio.gather(
                    _gen_response(), _gen_improved(), _gen_tips(),
                    return_exceptions=True,
                ),
                timeout=self.RESPONSE_NODE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[response_node] timed out after {self.RESPONSE_NODE_TIMEOUT}s")
            resp_obj, improved, tips = None, "", ""

        if isinstance(resp_obj, Exception) or resp_obj is None:
            coach_text = f"[{action.strategy.value.title()}]\n\n(unavailable)"
            last_q = ""
        else:
            strategy_label = action.strategy.value.title()
            coach_text = f"[{strategy_label}]\n\n{resp_obj.text}"
            last_q = resp_obj.text

        improved_text = improved if isinstance(improved, str) else ""
        language_tips = tips    if isinstance(tips,    str) else ""

        errors = analysis.pronunciation.mispronounced_words
        pronunciation_text = (
            "\n".join(
                f"• {w.word}  expected: {w.expected_ipa}  actual: {w.actual_ipa}  [{w.severity.value}]"
                for w in errors
            ) if errors else "✓ No pronunciation issues detected"
        )

        turn_num   = state.get("turn_number", 1)
        score      = arg.argument_score
        emoji      = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        show_wrapup = turn_num >= 3
        status = (
            f"✅ Turn {turn_num} complete  {emoji} Score: {score:.2f}"
            + (" — Choose an option below" if show_wrapup else " — Record your next argument above")
        )

        logger.info(f"[response_node] turn={turn_num} strategy={action.strategy.value}")
        return {
            "coach_text":          coach_text,
            "improved_text":       improved_text,
            "language_tips":       language_tips,
            "pronunciation_text":  pronunciation_text,
            "status_message":      status,
            "last_coach_question": last_q,
            "last_turn_intent":    state.get("turn_intent", ""),
        }

    # ── Node: update session ──────────────────────────────────────────────────

    async def _update_session_node(self, state: SpeakFlowState) -> dict:
        """Persist per-turn data into state for use by future turns."""
        analysis = state.get("turn_analysis")
        action   = state.get("coaching_action")

        prior_turns      = list(state.get("prior_turns")       or [])
        coaching_history = list(state.get("coaching_history")  or [])
        argument_scores  = list(state.get("argument_scores")   or [])
        prior_responses  = list(state.get("prior_responses")   or [])
        session_transcripts  = list(state.get("session_transcripts")  or [])
        session_language_tips = list(state.get("session_language_tips") or [])

        if analysis:
            prior_turns.append({"summary": analysis.argument.summary or state.get("transcript", "")})
            argument_scores.append(analysis.argument.argument_score)
            session_transcripts.append(state.get("transcript", ""))

        if action:
            coaching_history.append(action.strategy.value)

        lang_tips = state.get("language_tips", "")
        session_language_tips.append(lang_tips)

        coach_text = state.get("coach_text", "")
        if coach_text:
            prior_responses.append(state.get("last_coach_question", ""))
            if len(prior_responses) > 5:
                prior_responses = prior_responses[-5:]

        return {
            "prior_turns":          prior_turns[-10:],   # cap to last 10
            "coaching_history":     coaching_history,
            "argument_scores":      argument_scores,
            "prior_responses":      prior_responses,
            "session_transcripts":  session_transcripts,
            "session_language_tips": session_language_tips,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def ainvoke(self, input_state: dict, config: dict) -> SpeakFlowState:
        """
        Async entry point. Called once per student turn.

        Args:
            input_state: dict with at minimum: transcript, topic, user_position,
                         session_id, audio_path (optional)
            config:      {"configurable": {"thread_id": session_id}}
                         MemorySaver uses thread_id to persist/restore state.

        Returns:
            Final SpeakFlowState after all nodes have run.
        """
        return await self._graph.ainvoke(input_state, config=config)

    def get_session_state(self, config: dict) -> dict:
        """
        Read the latest checkpoint state for a given thread_id.
        Used by stop_session() to access session_transcripts, argument_scores,
        and session_language_tips without re-running the pipeline.

        Returns an empty dict if no checkpoint exists yet.
        """
        try:
            checkpoint = self._graph.get_state(config)
            return checkpoint.values if checkpoint else {}
        except Exception as e:
            logger.warning(f"[get_session_state] could not read checkpoint: {e}")
            return {}

    def get_graph_image(self):
        """Return Mermaid diagram string for documentation."""
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return None