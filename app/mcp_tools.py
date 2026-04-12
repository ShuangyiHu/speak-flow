"""
mcp_tools.py — SpeakFlow AI MCP Tool Layer
==========================================
Implements a lightweight in-process MCP (Model Context Protocol) tool server
that abstracts all tool access behind standardized schemas.

Design principles:
  - Each tool has a declared input_schema (JSON Schema) and a registered handler
  - pipeline.py calls tools via SpeakFlowMCPClient.call_tool(name, args) only
  - src/ modules are never imported directly by pipeline.py after this refactor
  - Tools are stateless — all state lives in SpeakFlowState (LangGraph)
  - In-process transport (no network hop) preserves existing latency profile

Tool registry:
  ┌──────────────────────────┬─────────────────────────────────────────────┐
  │ Tool name                │ Replaces                                    │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ analyze_argument_scores  │ pipeline._score_node → turn_analyzer        │
  │ analyze_argument_summary │ pipeline._summary_node → turn_analyzer      │
  │ analyze_pronunciation    │ pipeline._pronunciation_node → turn_analyzer │
  │ retrieve_evidence        │ pipeline._rag_node → rag_retriever          │
  │ generate_response        │ pipeline._response_node → response_generator│
  └──────────────────────────┴─────────────────────────────────────────────┘

Usage (from pipeline.py):
    from mcp_tools import SpeakFlowMCPClient

    mcp = SpeakFlowMCPClient(
        turn_analyzer=self._turn_analyzer,
        rag_retriever=self._rag_retriever,
        response_gen=self._response_gen,
        pipeline_client=self._pipeline_client,
        pipeline_model=self._pipeline_model,
    )

    result = await mcp.call_tool("analyze_argument_scores", {
        "transcript": "...",
        "topic": "...",
        "user_position": "for",
        "turn_number": 1,
    })
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional

from shared_types import (
    CoachingAction,
    CoachingStrategy,
    PronunciationResult,
    TurnInput,
)

logger = logging.getLogger(__name__)


# ── Tool schema definition ────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """Descriptor for a single MCP tool (mirrors MCP spec structure)."""
    name: str
    description: str
    input_schema: Dict[str, Any]        # JSON Schema object
    handler: Callable[..., Coroutine]   # async fn(args: dict) -> dict


# ── MCP Tool Server ───────────────────────────────────────────────────────────

class SpeakFlowMCPServer:
    """
    In-process MCP tool server.

    Registers tools with typed schemas and dispatches call_tool() requests.
    Intentionally mirrors the MCP server interface so this class could be
    wrapped with a real stdio/SSE transport in the future with minimal changes.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"[MCPServer] registered tool: {tool.name}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool descriptors (name + schema) — mirrors MCP list_tools."""
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call by name.

        Returns:
            dict with at least {"ok": bool, "result": Any}
            On error: {"ok": False, "error": str, "result": <fallback>}
        """
        if name not in self._tools:
            raise ValueError(f"Unknown MCP tool: '{name}'. "
                             f"Available: {list(self._tools.keys())}")
        tool = self._tools[name]
        try:
            result = await tool.handler(args)
            return {"ok": True, "result": result}
        except Exception as e:
            logger.error(f"[MCPServer] tool '{name}' failed: {type(e).__name__}: {e}")
            return {"ok": False, "error": str(e), "result": None}


# ── MCP Client (used by pipeline.py) ─────────────────────────────────────────

class SpeakFlowMCPClient:
    """
    Thin client that pipeline.py uses to call tools.

    On construction it builds and wires the MCP server with all tool handlers
    backed by the same src/ module instances that pipeline.py already owns.
    This preserves the single-instance pattern (no double initialisation).

    Args:
        turn_analyzer:   TurnAnalyzer instance (from pipeline)
        rag_retriever:   RAGRetriever instance (from pipeline)
        response_gen:    ResponseGenerator instance (from pipeline)
        pipeline_client: AsyncAnthropic client shared across pipeline (max_retries=1)
        pipeline_model:  Model string (from env)
    """

    def __init__(
        self,
        turn_analyzer: Any,
        rag_retriever: Any,
        response_gen: Any,
        pipeline_client: Any,
        pipeline_model: str,
    ) -> None:
        self._server = SpeakFlowMCPServer()
        self._register_all_tools(
            turn_analyzer, rag_retriever, response_gen,
            pipeline_client, pipeline_model,
        )

    # ── Public call interface ─────────────────────────────────────────────────

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a registered tool. Returns {"ok": bool, "result": ...}."""
        return await self._server.call_tool(name, args)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Expose tool catalog (useful for LangSmith tracing / debugging)."""
        return self._server.list_tools()

    # ── Tool registration ─────────────────────────────────────────────────────

    def _register_all_tools(
        self,
        turn_analyzer: Any,
        rag_retriever: Any,
        response_gen: Any,
        pipeline_client: Any,
        pipeline_model: str,
    ) -> None:

        # ── Tool 1: analyze_argument_scores ───────────────────────────────────
        async def _handle_analyze_scores(args: dict) -> dict:
            """
            Calls Claude for 4 numeric scores + boolean argument flags.
            Mirrors the logic previously inlined in pipeline._score_node().
            """
            transcript = args["transcript"]
            topic      = args["topic"]
            position   = args.get("user_position", "for")
            turn_num   = args.get("turn_number", 1)

            prompt = (
                f'Topic: {topic} | Position: {position}\n'
                f'Student said: "{transcript[:400]}"\n\n'
                'Score this debate turn. Return ONLY valid JSON, no markdown:\n'
                '{\n'
                '  "clarity_score": <0.0-1.0>,\n'
                '  "reasoning_score": <0.0-1.0>,\n'
                '  "depth_score": <0.0-1.0>,\n'
                '  "fluency_score_arg": <0.0-1.0>,\n'
                '  "argument_score": <0.0-1.0>,\n'
                '  "has_claim": <true|false>,\n'
                '  "has_reasoning": <true|false>,\n'
                '  "has_evidence": <true|false>,\n'
                '  "logical_gaps": [...],\n'
                '  "vocabulary_flags": [...]\n'
                '}'
            )
            response = await pipeline_client.messages.create(
                model=pipeline_model,
                max_tokens=300,
                system="You are a debate scoring engine. Return JSON only. No markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            logger.info(f"[tool:analyze_argument_scores] turn={turn_num} "
                        f"score={data.get('argument_score', 0):.2f}")
            return data

        self._server.register(MCPTool(
            name="analyze_argument_scores",
            description=(
                "Score a debate turn on 4 dimensions (clarity, reasoning, depth, fluency) "
                "and extract boolean argument structure flags (has_claim, has_reasoning, "
                "has_evidence). Returns numeric scores only — no feedback text."
            ),
            input_schema={
                "type": "object",
                "required": ["transcript", "topic"],
                "properties": {
                    "transcript":    {"type": "string", "description": "Student's spoken turn"},
                    "topic":         {"type": "string", "description": "Debate topic"},
                    "user_position": {"type": "string", "enum": ["for", "against"]},
                    "turn_number":   {"type": "integer"},
                },
            },
            handler=_handle_analyze_scores,
        ))

        # ── Tool 2: analyze_argument_summary ──────────────────────────────────
        async def _handle_analyze_summary(args: dict) -> dict:
            """
            Calls Claude for per-dimension feedback text + summary sentence.
            Mirrors the logic previously inlined in pipeline._summary_node().
            """
            transcript = args["transcript"]
            topic      = args["topic"]
            position   = args.get("user_position", "for")
            turn_num   = args.get("turn_number", 1)

            prompt = (
                f'Topic: {topic} | Position: {position}\n'
                f'Student said: "{transcript[:400]}"\n\n'
                'Write short coaching feedback. Each value: one encouraging phrase, max 10 words.\n'
                'JSON only:\n'
                '{"clarity_feedback":"...","reasoning_feedback":"...",'
                '"depth_feedback":"...","fluency_feedback":"...",'
                '"summary":"one encouraging sentence max 20 words"}'
            )
            response = await pipeline_client.messages.create(
                model=pipeline_model,
                max_tokens=200,
                system="JSON only. No markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            logger.info(f"[tool:analyze_argument_summary] turn={turn_num} done")
            return data

        self._server.register(MCPTool(
            name="analyze_argument_summary",
            description=(
                "Generate per-dimension coaching feedback text (clarity, reasoning, depth, "
                "fluency) and a one-sentence summary. Returns text only — no numeric scores."
            ),
            input_schema={
                "type": "object",
                "required": ["transcript", "topic"],
                "properties": {
                    "transcript":    {"type": "string"},
                    "topic":         {"type": "string"},
                    "user_position": {"type": "string", "enum": ["for", "against"]},
                    "turn_number":   {"type": "integer"},
                },
            },
            handler=_handle_analyze_summary,
        ))

        # ── Tool 3: analyze_pronunciation ────────────────────────────────────
        async def _handle_analyze_pronunciation(args: dict) -> dict:
            """
            Runs pronunciation analysis via TurnAnalyzer (MFA stub or real MFA).
            Mirrors the logic previously inlined in pipeline._pronunciation_node().
            Returns a dict representation of PronunciationResult.
            """
            turn_input = TurnInput(
                transcript    = args["transcript"],
                session_id    = args.get("session_id", ""),
                turn_number   = args.get("turn_number", 1),
                topic         = args.get("topic", ""),
                user_position = args.get("user_position", "for"),
                audio_path    = args.get("audio_path", ""),
                prior_turns   = args.get("prior_turns", []),
            )
            result: PronunciationResult = await turn_analyzer._analyze_pronunciation(turn_input)
            # Serialize to dict for uniform MCP return type
            return {
                "mispronounced_words": [
                    {
                        "word":         w.word,
                        "expected_ipa": w.expected_ipa,
                        "actual_ipa":   w.actual_ipa,
                        "severity":     w.severity.value if hasattr(w.severity, "value") else w.severity,
                    }
                    for w in result.mispronounced_words
                ],
                "fluency_score":    result.fluency_score,
                "target_phonemes":  result.target_phonemes,
                "_raw_result":      result,  # kept for merge_analysis_node
            }

        self._server.register(MCPTool(
            name="analyze_pronunciation",
            description=(
                "Run phoneme-level pronunciation analysis using MFA (or stub). "
                "Returns mispronounced words with IPA transcriptions and fluency score."
            ),
            input_schema={
                "type": "object",
                "required": ["transcript"],
                "properties": {
                    "transcript":    {"type": "string"},
                    "session_id":    {"type": "string"},
                    "turn_number":   {"type": "integer"},
                    "topic":         {"type": "string"},
                    "user_position": {"type": "string"},
                    "audio_path":    {"type": "string"},
                    "prior_turns":   {"type": "array"},
                },
            },
            handler=_handle_analyze_pronunciation,
        ))

        # ── Tool 4: retrieve_evidence ────────────────────────────────────────
        async def _handle_retrieve_evidence(args: dict) -> Any:
            """
            HyDE retrieval via RAGRetriever.
            Mirrors the logic previously inlined in pipeline._rag_node().
            Returns the RetrievalContext object directly (pipeline reads it as-is).
            """
            # Re-hydrate a minimal CoachingAction for the retriever's HyDE query
            pre_action = CoachingAction(
                strategy              = CoachingStrategy.PROBE,
                topic                 = args.get("topic", ""),
                user_position         = args.get("user_position", "for"),
                intent                = "",
                target_claim          = None,
                target_word           = None,
                target_phoneme        = None,
                argument_score        = args.get("argument_score", 0.5),
                pronunciation_score   = 1.0,
                difficulty_delta      = 0,
                turn_number           = args.get("turn_number", 1),
                prior_coach_responses = args.get("prior_coach_responses", []),
            )
            turn_analysis = args["_turn_analysis"]  # passed as Python object, not serialized
            ctx = await rag_retriever.retrieve(pre_action, turn_analysis)
            logger.info(f"[tool:retrieve_evidence] fallback={ctx.fallback_used} "
                        f"chunks={len(ctx.chunks)}")
            return ctx

        self._server.register(MCPTool(
            name="retrieve_evidence",
            description=(
                "Retrieve debate evidence chunks from ChromaDB using HyDE. "
                "Returns a RetrievalContext with ranked chunks and fallback flag."
            ),
            input_schema={
                "type": "object",
                "required": ["_turn_analysis"],
                "properties": {
                    "topic":                  {"type": "string"},
                    "user_position":          {"type": "string"},
                    "argument_score":         {"type": "number"},
                    "turn_number":            {"type": "integer"},
                    "prior_coach_responses":  {"type": "array"},
                    "_turn_analysis":         {
                        "type": "object",
                        "description": "TurnAnalysis Python object (in-process only)",
                    },
                },
            },
            handler=_handle_retrieve_evidence,
        ))

        # ── Tool 5: generate_response ────────────────────────────────────────
        async def _handle_generate_response(args: dict) -> dict:
            """
            Generates coach_text, improved_text, and language_tips via ResponseGenerator.
            Mirrors the logic previously inlined in pipeline._response_node().
            All three sub-calls run in parallel (asyncio.gather) inside ResponseGenerator.
            """
            import asyncio
            coaching_action  = args["_coaching_action"]   # CoachingAction object
            turn_analysis    = args["_turn_analysis"]     # TurnAnalysis object
            retrieval_context = args.get("_retrieval_context")

            from shared_types import ResponseRequest
            
            req = ResponseRequest(
                coaching_action   = coaching_action,
                topic             = args.get("topic", ""),
                user_position     = args.get("user_position", "for"),
                prior_responses   = args.get("prior_responses", []),
                turn_number       = args.get("turn_number", 1),
                retrieval_context = retrieval_context,
            )
            generated = await response_gen.generate_response(req)
            coach_text = generated.text

            improved_text = await response_gen.generate_improved_version(
                original_transcript = args.get("_turn_analysis").turn_input.transcript,
                topic               = args.get("topic", ""),
                user_position       = args.get("user_position", "for"),
                vocabulary_flags    = args.get("_turn_analysis").argument.vocabulary_flags,
            )
            
            language_tips = await response_gen.generate_language_tips(
                original_transcript  = args["_turn_analysis"].turn_input.transcript,
                improved_transcript  = improved_text,
                vocabulary_flags     = args["_turn_analysis"].argument.vocabulary_flags,
                clarity_feedback     = args["_turn_analysis"].argument.clarity_feedback,
                fluency_feedback     = args["_turn_analysis"].argument.fluency_feedback,
            )

            logger.info("[tool:generate_response] coach/improved/tips done")
            return {
                "coach_text":    coach_text,
                "improved_text": improved_text,
                "language_tips": language_tips,
            }

        self._server.register(MCPTool(
            name="generate_response",
            description=(
                "Generate the three output texts for a coaching turn: "
                "(1) coach_text — the coach's response to the student, "
                "(2) improved_text — a rewritten version of the student's argument, "
                "(3) language_tips — grammar/vocab tips. All generated in parallel."
            ),
            input_schema={
                "type": "object",
                "required": ["_coaching_action", "_turn_analysis"],
                "properties": {
                    "session_id":           {"type": "string"},
                    "topic":                {"type": "string"},
                    "user_position":        {"type": "string"},
                    "turn_number":          {"type": "integer"},
                    "coaching_history":     {"type": "array"},
                    "argument_scores":      {"type": "array"},
                    "prior_responses":      {"type": "array"},
                    "last_coach_question":  {"type": "string"},
                    "_coaching_action":     {
                        "type": "object",
                        "description": "CoachingAction Python object (in-process only)",
                    },
                    "_turn_analysis":       {
                        "type": "object",
                        "description": "TurnAnalysis Python object (in-process only)",
                    },
                    "_retrieval_context":   {
                        "type": "object",
                        "description": "RetrievalContext Python object (in-process only)",
                    },
                },
            },
            handler=_handle_generate_response,
        ))
