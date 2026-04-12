# SpeakFlow AI — Real-Time Adaptive Debate Coach

An AI-powered English debate coaching app for Chinese L2 learners. Students speak; the coach listens, scores, and responds — adapting in real time to argument quality, pronunciation, and session history.

---

## The Problem

Chinese students face three compounding challenges in English speaking:

- **Confidence gap** — years of test-prep culture discourage open-ended expression
- **Logic gap** — curricula rarely train structured argumentation (claim → reason → evidence)
- **Pronunciation gap** — critical for _gaokao_ listening/speaking exams, yet rarely drilled in context

SpeakFlow addresses all three through a single interaction: a real-time AI debate partner.

---

## Architecture

The project has three distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PRODUCT  (runtime, serves users)                     │
│                                                                 │
│  User Audio                                                     │
│      → Whisper (transcription)                                  │
│      → LangGraph Pipeline (pipeline.py)  ← planner/router only │
│           intent_node                                           │
│           ├── fan-out: score_node ‖ summary_node ‖ pronun_node  │
│           ├── merge_analysis_node                               │
│           ├── fan-out: coach_policy_node ‖ rag_node             │
│           ├── response_node                                     │
│           └── update_session_node → MemorySaver                 │
│                    │  each node calls tools via MCP client      │
│      → MCP Tool Layer (mcp_tools.py)  ← standardized interface │
│           tool: analyze_argument_scores                         │
│           tool: analyze_argument_summary                        │
│           tool: analyze_pronunciation                           │
│           tool: retrieve_evidence                               │
│           tool: generate_response                               │
│                    │  handlers delegate to src/ modules         │
│      → src/ modules (turn_analyzer, rag_retriever, …)          │
│      → Gradio UI  (app.py)                                      │
└─────────────────────────────────────────────────────────────────┘
           ↑ each module built by the layer below

┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — DEV TEAM  (build-time, generates the codebase)       │
│                                                                 │
│  Requirements spec (MODULES dict in dev_team/main.py)           │
│      → Architect Agent    (design spec)                         │
│      → Backend Engineer   (Python implementation)               │
│      → Code Reviewer      (APPROVE / REQUEST_CHANGES)           │
│      → Test Engineer      (pytest suite)                        │
│      → Frontend Engineer  (Gradio prototype)                    │
│               ↓                                                 │
│      → GitHub PR  (auto-merge if APPROVE)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer    | Component              | Technology                                |
| -------- | ---------------------- | ----------------------------------------- |
| Product  | Speech-to-text         | OpenAI Whisper                            |
| Product  | Argument analysis      | Claude claude-sonnet-4-5 (Anthropic SDK)  |
| Product  | Pronunciation analysis | Montreal Forced Aligner (MFA) — stub mode |
| Product  | Agent orchestration    | LangGraph StateGraph + MemorySaver        |
| Product  | Tool abstraction       | MCP (Model Context Protocol) — in-process |
| Product  | RAG retrieval          | ChromaDB + sentence-transformers (HyDE)   |
| Product  | API server             | FastAPI + WebSocket (planned)             |
| Product  | Frontend prototype     | Gradio                                    |
| Dev Team | Agent framework        | CrewAI                                    |
| Dev Team | GitHub automation      | PyGithub                                  |
| Infra    | LLM observability      | LangSmith                                 |

---

## LangGraph Pipeline

`pipeline.py` is the orchestration core. It replaces the manual `asyncio.gather` chains from earlier versions with a typed `StateGraph` that makes parallelism, routing, and state management explicit.

```
START
  └─ intent_node  (no API — keyword routing)
       ├─ META_QUESTION → meta_handler_node → END
       ├─ OFF_TOPIC     → off_topic_node    → END
       └─ DEBATE_STATEMENT
            └─ fan-out (Send API — true parallelism):
                 ├─ score_node         (→ MCP: analyze_argument_scores,  ~1.5s)
                 ├─ summary_node       (→ MCP: analyze_argument_summary, ~2s)
                 └─ pronunciation_node (→ MCP: analyze_pronunciation)
            └─ merge_analysis_node  (assembles TurnAnalysis — no API)
            └─ fan-out (parallel):
                 ├─ coach_policy_node  (rule-based, no API)
                 └─ rag_node           (→ MCP: retrieve_evidence)
            └─ response_node       (→ MCP: generate_response)
            └─ update_session_node (appends to MemorySaver checkpoint)
            └─ END
```

**Key design decisions:**

- `score_node` and `summary_node` split the original single `_analyze_argument()` Claude call. `CoachPolicyAgent` only needs scores to decide strategy, so it can start as soon as `score_node` completes (~1.5s) without waiting for feedback text (~2s). Net saving: ~1.5–2s per turn.
- All session state (`prior_turns`, `coaching_history`, `argument_scores`, etc.) lives in `SpeakFlowState` and is persisted by `MemorySaver`, keyed by `thread_id`.
- `reset_session()` rotates to a new `thread_id` (UUID-based) for a clean slate.

---

## MCP Tool Layer

`mcp_tools.py` introduces an in-process [Model Context Protocol](https://modelcontextprotocol.io/) tool server that standardizes all tool access behind typed schemas.

### Why MCP?

Before this layer, `pipeline.py` nodes directly imported and called `src/` modules. This created **code-level coupling**: changing a tool's implementation required auditing the pipeline. After the refactor:

- Each pipeline node calls `await self._mcp.call_tool(name, args)` only
- Each tool has a declared **JSON Schema** (`input_schema`) — the contract between planner and executor
- `src/` modules are never imported by `pipeline.py` directly
- **Swapping a tool implementation** (e.g. replacing Whisper with a different ASR) requires changes only inside `mcp_tools.py`, not in `pipeline.py`

### Transport choice

The MCP server runs **in-process** (no network hop). This preserves the existing latency profile — adding a network transport would add ~5–20ms of serialization overhead per tool call, multiplied across 3 parallel fan-out calls. The `SpeakFlowMCPServer` / `SpeakFlowMCPClient` classes mirror the MCP spec interface, so switching to stdio or SSE transport later requires only a transport swap, not an interface change.

### Tool registry

| Tool name                  | Replaces (before refactor)                         |
| -------------------------- | -------------------------------------------------- |
| `analyze_argument_scores`  | `pipeline._score_node` → `turn_analyzer` (direct) |
| `analyze_argument_summary` | `pipeline._summary_node` → `turn_analyzer` (direct)|
| `analyze_pronunciation`    | `pipeline._pronunciation_node` → `turn_analyzer`  |
| `retrieve_evidence`        | `pipeline._rag_node` → `rag_retriever` (direct)   |
| `generate_response`        | `pipeline._response_node` → `response_gen` (direct)|

---

## Repository Structure

```
speakflow/
│
├── app/
│   ├── app.py              # Gradio UI — session lifecycle only
│   ├── pipeline.py         # LangGraph StateGraph — planner/router; calls tools via MCP
│   └── mcp_tools.py        # MCP tool server — 5 tools, typed schemas, src/ handlers
│
├── src/                    # Product modules — generated by Dev Team
│   ├── shared_types.py     # Single source of truth for all dataclasses/enums
│   ├── turn_analyzer.py    # Argument scoring + pronunciation (MFA stub)
│   ├── coach_policy.py     # Rule-based strategy selection
│   ├── response_generator.py  # Claude-powered natural language generation
│   └── rag_retriever.py    # HyDE retrieval from ChromaDB debate knowledge base
│
├── tests/
│   ├── test_pipeline.py    # Smoke test: all graph paths + MemorySaver persistence
│   ├── test_turn_analyzer.py
│   └── test_coach_policy.py
│
├── dev_team/               # CrewAI multi-agent development pipeline
│   ├── crew.py
│   ├── main.py             # Runner + MODULES registry
│   ├── github_integration.py
│   └── config/
│       ├── agents.yaml
│       └── tasks.yaml
│
├── .env                    # Environment variables (never commit)
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/ShuangyiHu/speak-flow.git
cd speak-flow
pip install anthropic langgraph langchain-anthropic gradio chromadb \
            sentence-transformers openai-whisper python-dotenv crewai PyGithub
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

```bash
# .env.example

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5

# LangSmith observability (optional but recommended)
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=speakflow

# Dev Team: GitHub integration
GITHUB_TOKEN=...
GITHUB_REPO=ShuangyiHu/speak-flow

# Development flags
USE_STUB_MFA=True
WHISPER_MODEL=base
```

### 3. Run the pipeline smoke test

```bash
cd app
python test_pipeline.py
```

All 5 test sections should show ✅. This validates the full LangGraph graph including fan-out/fan-in, MemorySaver persistence, and all routing paths.

### 4. Run the Gradio app

```bash
cd app
python app.py
```

### 5. Run the Dev Team (optional — generates new modules)

```bash
cd dev_team
python main.py --module turn_analyzer
```

---

## Project Status

| Component                | Status      | Notes                                          |
| ------------------------ | ----------- | ---------------------------------------------- |
| `turn_analyzer.py`       | ✅ Complete | 4-dimension scoring, MFA stub                  |
| `coach_policy.py`        | ✅ Complete | Rule-based strategy selection                  |
| `response_generator.py`  | ✅ Complete | Coach text, improved version, tips             |
| `rag_retriever.py`       | ✅ Complete | HyDE + ChromaDB, fallback stubs                |
| `shared_types.py`        | ✅ Complete | Single source of truth                         |
| `pipeline.py`            | ✅ Complete | LangGraph StateGraph; calls tools via MCP      |
| `mcp_tools.py`           | ✅ Complete | MCP tool server; 5 tools with typed schemas    |
| `app.py`                 | ✅ Complete | Gradio UI, pipeline-integrated                 |
| MFA integration          | 🔄 Stub     | USE_STUB_MFA=True; real MFA planned            |
| React + FastAPI frontend | ⏳ Planned  | Replace Gradio for production                  |
| `session_evaluator.py`   | ⏳ Planned  | Multi-round trend analysis                     |

---

## Background

This project demonstrates two things on a single portfolio:

1. **AI product development** — designing a real-time coaching pipeline with LangGraph orchestration, MCP tool abstraction, RAG retrieval, audio processing, and structured feedback
2. **AI-driven development** — using a CrewAI agent team to build the product itself, with traceable GitHub PR history for every module

The target users are Chinese high school students preparing for _gaokao_ English speaking exams and beyond — students who have ideas but have never been given the space to argue in English.
