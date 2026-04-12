"""
pipeline.py — SpeakFlow AI LangGraph Orchestration Layer  (MCP refactor)
=========================================================================
Replaces the manual asyncio.gather chains in app.py with a typed StateGraph.

Graph topology (unchanged — MCP refactor does NOT change the graph structure):
    START
      └─ intent_node          (no API — keyword routing)
           ├─ META_QUESTION  → meta_handler_node → END
           ├─ OFF_TOPIC      → off_topic_node    → END
           └─ DEBATE_STATEMENT
                └─ fan-out via Send API (true parallelism):
                     ├─ score_node          (→ MCP: analyze_argument_scores)
                     ├─ summary_node        (→ MCP: analyze_argument_summary)
                     └─ pronunciation_node  (→ MCP: analyze_pronunciation)
                └─ merge_analysis_node  (assemble TurnAnalysis — unchanged)
                └─ fan-out (parallel):
                     ├─ coach_policy_node  (rule-based — unchanged)
                     └─ rag_node           (→ MCP: retrieve_evidence)
                └─ response_node        (→ MCP: generate_response)
                └─ update_session_node  (unchanged)
                └─ END

MCP refactor summary:
  - __init__: adds SpeakFlowMCPClient construction (3 lines)
  - score_node, summary_node, pronunciation_node, rag_node, response_node:
    each replaced with a single mcp.call_tool() dispatch
  - All other nodes (intent, meta, off_topic, merge, coach_policy,
    update_session) are 100% unchanged
  - src/ modules are no longer imported or called directly by any node;
    they are only referenced inside mcp_tools.py handlers

Key design decisions (unchanged from previous version):
  - score_node and summary_node split the original single _analyze_argument()
    Claude call, enabling CoachPolicyAgent to start as soon as scores are
    ready (~1.5s) without waiting for feedback text (~2s).
  - All session state lives in SpeakFlowState and is persisted by MemorySaver,
    keyed by thread_id. app.py holds no session state of its own.
  - reset_session() rotates to a new thread_id (UUID-based) for a clean slate.
"""

import asyncio
import importlib
import json
import logging
import os
import re as _re
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
    _score_result:     Annotated[Optional[dict],   _keep_last]
    _summary_result:   Annotated[Optional[dict],   _keep_last]
    _pronunciation_result: Annotated[Optional[PronunciationResult], _keep_last]

    # ── Pipeline outputs (written by nodes, read by app.py) ───────────────────
    turn_intent:       Annotated[str,              _keep_last]
    turn_analysis:     Annotated[Optional[TurnAnalysis],    _keep_last]
    coaching_action:   Annotated[Optional[CoachingAction],  _keep_last]
    retrieval_context: Annotated[Any,              _keep_last]
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

        # Dedicated pipeline client: max_retries=1 prevents the 20s+ blowout
        # from the default max_retries=2 exponential backoff policy.
        self._pipeline_client = AsyncAnthropic(
            api_key=api_key,
            max_retries=1,
            timeout=float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30")),
        )
        self._pipeline_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        # Unify: replace each module's internal Anthropic client with
        # _pipeline_client so ALL API calls share the same retry policy.
        self._response_gen._client    = self._pipeline_client
        self._rag_retriever._anthropic = self._pipeline_client

        # ── MCP tool layer (NEW) ───────────────────────────────────────────────
        # Builds the in-process MCP server and wires all tool handlers.
        # pipeline nodes call tools via self._mcp.call_tool(name, args) only;
        # no node imports src/ modules directly after this refactor.
        from mcp_tools import SpeakFlowMCPClient
        self._mcp = SpeakFlowMCPClient(
            turn_analyzer  = self._turn_analyzer,
            rag_retriever  = self._rag_retriever,
            response_gen   = self._response_gen,
            pipeline_client = self._pipeline_client,
            pipeline_model  = self._pipeline_model,
        )
        logger.info(f"[pipeline] MCP tools registered: "
                    f"{[t['name'] for t in self._mcp.list_tools()]}")

        self._graph = self._build_graph()

    # ── Graph builder (unchanged) ─────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(SpeakFlowState)

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

        g.add_edge(START, "intent_node")
        g.add_conditional_edges("intent_node", self._route_by_intent)
        g.add_edge("meta_handler_node",   END)
        g.add_edge("off_topic_node",      END)

        g.add_edge("score_node",         "merge_analysis_node")
        g.add_edge("summary_node",       "merge_analysis_node")
        g.add_edge("pronunciation_node", "merge_analysis_node")

        g.add_edge("merge_analysis_node",  "coach_policy_node")
        g.add_edge("merge_analysis_node",  "rag_node")
        g.add_edge("coach_policy_node",    "response_node")
        g.add_edge("rag_node",             "response_node")

        g.add_edge("response_node",        "update_session_node")
        g.add_edge("update_session_node",  END)

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

    # ── Routing (unchanged) ───────────────────────────────────────────────────

    def _route_by_intent(self, state: SpeakFlowState):
        intent = state.get("turn_intent", TurnIntent.DEBATE_STATEMENT)
        if intent == TurnIntent.META_QUESTION:
            return "meta_handler_node"
        if intent == TurnIntent.OFF_TOPIC:
            return "off_topic_node"
        return [
            Send("score_node",         state),
            Send("summary_node",       state),
            Send("pronunciation_node", state),
        ]

    # ── Node: intent (unchanged) ──────────────────────────────────────────────

    async def _intent_node(self, state: SpeakFlowState) -> dict:
        transcript = state.get("transcript", "")
        intent = self._turn_analyzer._detect_intent(transcript)
        current = state.get("turn_number") or 0
        turn_number = current + 1 if intent == TurnIntent.DEBATE_STATEMENT else current
        logger.info(f"[intent_node] turn={turn_number} intent={intent}")
        return {"turn_intent": intent, "turn_number": turn_number}

    # ── Node: meta handler (unchanged) ───────────────────────────────────────

    async def _meta_handler_node(self, state: SpeakFlowState) -> dict:
        transcript = state.get("transcript", "")
        topic      = state.get("topic", "")
        position   = state.get("user_position", "for")
        prompt = (
            f'Topic: "{topic}". Student is arguing {position}.\n'
            f'Student said: "{transcript[:300]}"\n\n'
            'This is a meta-question or pushback, not a debate argument. '
            'Respond briefly (1–2 sentences) to acknowledge the question, '
            'then redirect them to continue their debate argument. '
            'Do not score or coach — just answer and redirect.'
        )
        try:
            response = await self._pipeline_client.messages.create(
                model=self._pipeline_model,
                max_tokens=120,
                system="You are a helpful debate coach. Be concise and encouraging.",
                messages=[{"role": "user", "content": prompt}],
            )
            coach_text = response.content[0].text.strip()
        except Exception as e:
            logger.error(f"[meta_handler_node] {e}")
            coach_text = "Good question! Let's keep going — try building on your main argument."
        return {
            "coach_text":     coach_text,
            "improved_text":  "",
            "language_tips":  "",
            "status_message": "💬 Answered your question — keep debating!",
        }

    # ── Node: off topic (unchanged) ───────────────────────────────────────────

    async def _off_topic_node(self, state: SpeakFlowState) -> dict:
        topic = state.get("topic", "the debate topic")
        return {
            "coach_text":     f"Let's stay focused on: '{topic}'. What's your argument?",
            "improved_text":  "",
            "language_tips":  "",
            "status_message": "↩ Off-topic — please respond to the debate.",
        }

    # ── Node: score  ← MCP refactored ────────────────────────────────────────

    async def _score_node(self, state: SpeakFlowState) -> dict:
        """
        Calls MCP tool 'analyze_argument_scores' to get 4 numeric scores +
        boolean argument flags via Claude.

        BEFORE (direct): called self._pipeline_client.messages.create() inline
        AFTER  (MCP):    calls self._mcp.call_tool("analyze_argument_scores", args)
        """
        call = await self._mcp.call_tool("analyze_argument_scores", {
            "transcript":    state.get("transcript", ""),
            "topic":         state.get("topic", ""),
            "user_position": state.get("user_position", "for"),
            "turn_number":   state.get("turn_number", 1),
        })
        if call["ok"]:
            return {"_score_result": call["result"]}
        logger.error(f"[score_node] MCP tool failed: {call.get('error')}")
        return {"_score_result": {
            "clarity_score": 0.0, "reasoning_score": 0.0,
            "depth_score": 0.0,   "fluency_score_arg": 0.0,
            "argument_score": 0.0, "has_claim": False,
            "has_reasoning": False, "has_evidence": False,
            "logical_gaps": [], "vocabulary_flags": [],
        }}

    # ── Node: summary  ← MCP refactored ──────────────────────────────────────

    async def _summary_node(self, state: SpeakFlowState) -> dict:
        """
        Calls MCP tool 'analyze_argument_summary' for per-dimension feedback
        text + summary sentence via Claude.

        BEFORE (direct): called self._pipeline_client.messages.create() inline
        AFTER  (MCP):    calls self._mcp.call_tool("analyze_argument_summary", args)
        """
        call = await self._mcp.call_tool("analyze_argument_summary", {
            "transcript":    state.get("transcript", ""),
            "topic":         state.get("topic", ""),
            "user_position": state.get("user_position", "for"),
            "turn_number":   state.get("turn_number", 1),
        })
        if call["ok"]:
            return {"_summary_result": call["result"]}
        logger.error(f"[summary_node] MCP tool failed: {call.get('error')}")
        return {"_summary_result": {
            "clarity_feedback":   "Keep your position clear.",
            "reasoning_feedback": "Try to add a reason.",
            "depth_feedback":     "Add an example next time.",
            "fluency_feedback":   "Good effort with your English.",
            "summary":            "Keep going — you're building your argument well.",
        }}

    # ── Node: pronunciation  ← MCP refactored ────────────────────────────────

    async def _pronunciation_node(self, state: SpeakFlowState) -> dict:
        """
        Calls MCP tool 'analyze_pronunciation' to run MFA (stub or real).

        BEFORE (direct): built TurnInput and called self._turn_analyzer directly
        AFTER  (MCP):    calls self._mcp.call_tool("analyze_pronunciation", args)
                         and extracts the _raw_result PronunciationResult object
        """
        call = await self._mcp.call_tool("analyze_pronunciation", {
            "transcript":    state.get("transcript", ""),
            "session_id":    state.get("session_id", ""),
            "turn_number":   state.get("turn_number", 1),
            "topic":         state.get("topic", ""),
            "user_position": state.get("user_position", "for"),
            "audio_path":    state.get("audio_path", ""),
            "prior_turns":   state.get("prior_turns") or [],
        })
        if call["ok"]:
            # The tool returns the PronunciationResult object under "_raw_result"
            # for merge_analysis_node (which expects the typed object, not a dict).
            return {"_pronunciation_result": call["result"]["_raw_result"]}
        logger.error(f"[pronunciation_node] MCP tool failed: {call.get('error')}")
        return {"_pronunciation_result":
                self._turn_analyzer._create_default_pronunciation_result()}

    # ── Node: merge analysis (unchanged) ─────────────────────────────────────

    async def _merge_analysis_node(self, state: SpeakFlowState) -> dict:
        """
        Fan-in: assemble TurnAnalysis from the three parallel partial results.
        Completely unchanged — it just reads _score_result, _summary_result,
        _pronunciation_result from state; it doesn't care who wrote them.
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
        arg_score = float(score_data.get("argument_score",
                    round(0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency, 3)))

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

    # ── Node: coach policy (unchanged) ────────────────────────────────────────

    async def _coach_policy_node(self, state: SpeakFlowState) -> dict:
        """Select coaching strategy — rule-based, no API call. Completely unchanged."""
        analysis = state.get("turn_analysis")
        if analysis is None:
            return {"coaching_action": None}

        arg = analysis.argument
        scores = {
            "Clarity":   arg.clarity_score,
            "Reasoning": arg.reasoning_score,
            "Depth":     arg.depth_score,
            "Fluency":   arg.fluency_score_arg,
        }
        weakest = [k for k, v in sorted(scores.items(), key=lambda x: x[1]) if v < 0.6][:2]
        dim_lines = [f"  {k}: {v:.2f}" for k, v in scores.items()]

        ctx = SessionContext(
            session_id          = state.get("session_id", ""),
            topic               = state.get("topic", ""),
            user_position       = state.get("user_position", "for"),
            turn_number         = state.get("turn_number", 1),
            coaching_history    = [
                CoachingStrategy(s) for s in (state.get("coaching_history") or [])
                if s in [e.value for e in CoachingStrategy]
            ],
            argument_scores     = list(state.get("argument_scores") or []),
            last_coach_question = state.get("last_coach_question", ""),
        )

        action = await self._coach.decide(analysis, ctx)
        action.intent = "\n".join([
            f"Student just said: \"{(state.get('transcript') or '')[:300]}\"",
            "Scores:", *dim_lines,
            f"Coach summary: {arg.summary}",
            f"Weakest areas: {', '.join(weakest) if weakest else 'none — strong'}",
            f"Coach last asked: \"{state.get('last_coach_question', '')}\"",
            "Do NOT repeat or rephrase the coach's last question.",
        ])

        logger.info(f"[coach_policy_node] strategy={action.strategy.value}")
        return {"coaching_action": action}

    # ── Node: RAG  ← MCP refactored ──────────────────────────────────────────

    async def _rag_node(self, state: SpeakFlowState) -> dict:
        """
        Calls MCP tool 'retrieve_evidence' for HyDE retrieval from ChromaDB.

        BEFORE (direct): built CoachingAction and called self._rag_retriever directly
        AFTER  (MCP):    calls self._mcp.call_tool("retrieve_evidence", args)
                         passing the TurnAnalysis object under "_turn_analysis"
        """
        analysis = state.get("turn_analysis")
        if analysis is None:
            return {"retrieval_context": None}

        call = await self._mcp.call_tool("retrieve_evidence", {
            "topic":                  state.get("topic", ""),
            "user_position":          state.get("user_position", "for"),
            "argument_score":         analysis.argument.argument_score,
            "turn_number":            state.get("turn_number", 1),
            "prior_coach_responses":  list(state.get("prior_responses") or [])[-3:],
            "_turn_analysis":         analysis,   # passed as Python object
        })
        if call["ok"]:
            ctx = call["result"]
            logger.info(f"[rag_node] fallback={ctx.fallback_used} chunks={len(ctx.chunks)}")
            return {"retrieval_context": ctx}
        logger.error(f"[rag_node] MCP tool failed: {call.get('error')}")
        return {"retrieval_context": None}

    # ── Node: response  ← MCP refactored ─────────────────────────────────────

    RESPONSE_NODE_TIMEOUT = 12.0

    async def _response_node(self, state: SpeakFlowState) -> dict:
        """
        Calls MCP tool 'generate_response' for coach_text, improved_text,
        and language_tips (all three generated in parallel inside the tool).

        BEFORE (direct): called self._response_gen.generate_* directly with
                         asyncio.gather() and a manual RESPONSE_NODE_TIMEOUT
        AFTER  (MCP):    calls self._mcp.call_tool("generate_response", args)
                         wrapped in asyncio.wait_for for the same timeout budget
        """
        coaching_action   = state.get("coaching_action")
        turn_analysis     = state.get("turn_analysis")
        retrieval_context = state.get("retrieval_context")

        if coaching_action is None or turn_analysis is None:
            return {
                "coach_text":     "Great effort! Keep building your argument.",
                "improved_text":  "",
                "language_tips":  "",
                "status_message": "⚠️ Missing analysis — using fallback response.",
            }

        try:
            call = await asyncio.wait_for(
                self._mcp.call_tool("generate_response", {
                    "session_id":          state.get("session_id", ""),
                    "topic":               state.get("topic", ""),
                    "user_position":       state.get("user_position", "for"),
                    "turn_number":         state.get("turn_number", 1),
                    "coaching_history":    list(state.get("coaching_history") or []),
                    "argument_scores":     list(state.get("argument_scores") or []),
                    "prior_responses":     list(state.get("prior_responses") or [])[-3:],
                    "last_coach_question": state.get("last_coach_question", ""),
                    "_coaching_action":    coaching_action,    # Python object
                    "_turn_analysis":      turn_analysis,      # Python object
                    "_retrieval_context":  retrieval_context,  # Python object
                }),
                timeout=self.RESPONSE_NODE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[response_node] MCP tool timed out after {self.RESPONSE_NODE_TIMEOUT}s")
            return {
                "coach_text":     "Good effort! Think about adding more evidence to support your claim.",
                "improved_text":  "",
                "language_tips":  "",
                "status_message": "⚠️ Response generation timed out.",
            }

        if not call["ok"]:
            logger.error(f"[response_node] MCP tool failed: {call.get('error')}")
            return {
                "coach_text":     "Good effort! Keep developing your argument.",
                "improved_text":  "",
                "language_tips":  "",
                "status_message": "⚠️ Response generation failed.",
            }

        result = call["result"]
        turn_num = state.get("turn_number", 1)
        wrapup   = turn_num >= 3

        return {
            "coach_text":     result["coach_text"],
            "improved_text":  result["improved_text"],
            "language_tips":  result["language_tips"],
            "status_message": (
                "🏁 Great session! Review your progress below."
                if wrapup else
                f"✅ Turn {turn_num} complete."
            ),
        }

    # ── Node: update session (unchanged) ─────────────────────────────────────

    async def _update_session_node(self, state: SpeakFlowState) -> dict:
        """Append scores and history to persisted state. Completely unchanged."""
        analysis = state.get("turn_analysis")
        action   = state.get("coaching_action")

        new_scores  = list(state.get("argument_scores") or [])
        new_history = list(state.get("coaching_history") or [])
        new_prior   = list(state.get("prior_turns") or [])
        new_responses = list(state.get("prior_responses") or [])
        new_transcripts = list(state.get("session_transcripts") or [])
        new_tips    = list(state.get("session_language_tips") or [])

        if analysis:
            new_scores.append(analysis.argument.argument_score)
            new_prior.append({
                "transcript":     analysis.turn_input.transcript,
                "argument_score": analysis.argument.argument_score,
                "has_claim":      analysis.argument.has_claim,
                "summary":        analysis.argument.summary,
            })
            new_transcripts.append(analysis.turn_input.transcript)

        if action:
            new_history.append(action.strategy.value)
        coach_text = state.get("coach_text", "")
        if coach_text:
            new_responses.append(coach_text)

        tips = state.get("language_tips", "")
        if tips:
            new_tips.append(tips)

        last_q = ""
        if coach_text and "?" in coach_text:
            sentences = [s.strip() for s in coach_text.split(".") if "?" in s]
            last_q = sentences[-1] if sentences else ""

        logger.info(f"[update_session_node] turn={state.get('turn_number')} "
                    f"scores_stored={len(new_scores)}")
        return {
            "argument_scores":      new_scores,
            "coaching_history":     new_history,
            "prior_turns":          new_prior,
            "prior_responses":      new_responses,
            "last_coach_question":  last_q,
            "last_turn_intent":     state.get("turn_intent", ""),
            "session_transcripts":  new_transcripts,
            "session_language_tips": new_tips,
        }

    # ── Public interface (unchanged) ──────────────────────────────────────────

    async def ainvoke(self, state: dict, config: dict) -> dict:
        result = await self._graph.ainvoke(state, config=config)
        return result

    def get_graph_image(self) -> Optional[str]:
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return None

    def get_session_state(self, config: dict) -> dict:
        try:
            checkpoint = self._graph.get_state(config)
            return checkpoint.values if checkpoint else {}
        except Exception as e:
            logger.error(f"[get_session_state] {e}")
            return {}