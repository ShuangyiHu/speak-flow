"""
test_pipeline.py — Standalone smoke test for pipeline.py
=========================================================
Run from the app/ directory (same level as pipeline.py):

    cd speak-flow/app
    python test_pipeline.py

What this tests:
    1. Graph builds without errors
    2. DEBATE_STATEMENT path — full fan-out/fan-in through all nodes
    3. META_QUESTION path — routes to meta_handler_node, skips analysis
    4. OFF_TOPIC path — routes to off_topic_node, skips analysis
    5. Second turn — verifies MemorySaver restores prior_turns / coaching_history
    6. Third turn — verifies turn_number >= 3 triggers wrapup status

Prints timing per node so you can see the score/summary split in action.
"""

import asyncio
import os
import time
import sys

# ── Make sure we can import from the same directory ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import SpeakFlowPipeline

SESSION_ID = "test-session-001"
CONFIG     = {"configurable": {"thread_id": SESSION_ID}}

TOPIC    = "Social media does more harm than good"
POSITION = "for"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    print(f"  {status}  {label}" + (f"  →  {detail}" if detail else ""))
    if not condition:
        print(f"       *** FAIL ***")


async def run_turn(pipeline: SpeakFlowPipeline, transcript: str, label: str) -> dict:
    print(f"\n  [{label}]")
    print(f"  transcript: \"{transcript[:80]}\"")
    t0 = time.time()
    result = await pipeline.ainvoke({
        "transcript":   transcript,
        "topic":        TOPIC,
        "user_position": POSITION,
        "session_id":   SESSION_ID,
        "audio_path":   "",
    }, config=CONFIG)
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.2f}s total")
    return result


async def main():
    print("\n🔧 SpeakFlow Pipeline Smoke Test")
    print(f"   topic:    {TOPIC}")
    print(f"   position: {POSITION}")

    # ── Init ──────────────────────────────────────────────────────────────────
    section("0. Pipeline init")
    t0 = time.time()
    try:
        pipeline = SpeakFlowPipeline()
        print(f"  ✅  Pipeline built in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  ❌  Pipeline init failed: {e}")
        raise

    # Optional: print Mermaid graph
    mermaid = pipeline.get_graph_image()
    if mermaid:
        print("\n  --- Mermaid graph (copy to https://mermaid.live) ---")
        print(mermaid)

    # ── Test 1: OFF_TOPIC ─────────────────────────────────────────────────────
    section("1. OFF_TOPIC path  (transcript too short)")
    result = await run_turn(pipeline, "I agree.", "off_topic")

    check("turn_intent == OFF_TOPIC",
          result.get("turn_intent") == "off_topic",
          str(result.get("turn_intent")))
    check("coach_text non-empty",
          bool(result.get("coach_text")),
          repr(result.get("coach_text", ""))[:80])
    check("turn_analysis is None",
          result.get("turn_analysis") is None,
          "correctly skipped analysis")

    # ── Test 2: META_QUESTION ─────────────────────────────────────────────────
    section("2. META_QUESTION path")
    result = await run_turn(pipeline,
        "I already said that social media is harmful. What do you mean?",
        "meta_question")

    check("turn_intent == META_QUESTION",
          result.get("turn_intent") == "meta_question",
          str(result.get("turn_intent")))
    check("coach_text contains [Coach]",
          "[Coach]" in (result.get("coach_text") or ""),
          repr(result.get("coach_text", ""))[:80])
    check("turn_analysis is None",
          result.get("turn_analysis") is None,
          "correctly skipped analysis")

    # ── Test 3: Turn 1 — full DEBATE_STATEMENT path ───────────────────────────
    section("3. Turn 1 — DEBATE_STATEMENT  (full fan-out/fan-in)")
    result = await run_turn(pipeline,
        "I believe social media does more harm than good "
        "because it spreads misinformation quickly. "
        "For example, during the pandemic many people believed false cures.",
        "turn_1")

    analysis = result.get("turn_analysis")
    check("turn_analysis is not None",    analysis is not None)
    check("turn_intent == debate_statement",
          result.get("turn_intent") == "debate_statement",
          str(result.get("turn_intent")))

    if analysis:
        arg = analysis.argument
        check("argument_score in [0,1]",
              0.0 <= arg.argument_score <= 1.0,
              f"{arg.argument_score:.2f}")
        check("has_claim is bool",        isinstance(arg.has_claim, bool),  str(arg.has_claim))
        check("summary non-empty",        bool(arg.summary),                repr(arg.summary[:60]))
        check("clarity_feedback present", bool(arg.clarity_feedback),       repr(arg.clarity_feedback[:40]))
        pron = analysis.pronunciation
        check("fluency_score in [0,1]",   0.0 <= pron.fluency_score <= 1.0, f"{pron.fluency_score:.2f}")

    check("coach_text non-empty",     bool(result.get("coach_text")),    repr(result.get("coach_text","")[:80]))
    check("improved_text non-empty",  bool(result.get("improved_text")), repr(result.get("improved_text","")[:60]))
    check("language_tips non-empty",  bool(result.get("language_tips")), repr(result.get("language_tips","")[:60]))
    check("status_message non-empty", bool(result.get("status_message")))
    check("turn_number == 1",         result.get("turn_number") == 1,    str(result.get("turn_number")))

    coaching_action = result.get("coaching_action")
    check("coaching_action present",  coaching_action is not None)
    if coaching_action:
        check("strategy is valid CoachingStrategy",
              coaching_action.strategy.value in
              ["PROBE","CHALLENGE","REDIRECT","PRAISE","CORRECT_PRONUNCIATION"],
              coaching_action.strategy.value)

    print(f"\n  📊 Scores: "
          f"clarity={analysis.argument.clarity_score:.2f}  "
          f"reasoning={analysis.argument.reasoning_score:.2f}  "
          f"depth={analysis.argument.depth_score:.2f}  "
          f"fluency={analysis.argument.fluency_score_arg:.2f}  "
          f"→ total={analysis.argument.argument_score:.2f}" if analysis else "")
    print(f"  🎓 Coach strategy: {coaching_action.strategy.value if coaching_action else 'N/A'}")
    print(f"  💬 Coach text: {result.get('coach_text','')[:100]}")

    # ── Test 4: Turn 2 — MemorySaver state persistence ────────────────────────
    section("4. Turn 2 — MemorySaver: prior_turns restored")
    result2 = await run_turn(pipeline,
        "Social media also causes addiction and mental health problems, "
        "especially among teenagers who spend hours scrolling every day.",
        "turn_2")

    check("turn_number == 2",
          result2.get("turn_number") == 2,
          str(result2.get("turn_number")))
    check("prior_turns has 1 entry from turn 1",
          len(result2.get("prior_turns") or []) >= 1,
          f"len={len(result2.get('prior_turns') or [])}")
    check("argument_scores has 1 entry from turn 1",
          len(result2.get("argument_scores") or []) >= 1,
          f"scores={result2.get('argument_scores')}")
    check("coaching_history has 1 entry from turn 1",
          len(result2.get("coaching_history") or []) >= 1,
          f"history={result2.get('coaching_history')}")
    check("status has 'next argument'",
          "Record your next argument" in (result2.get("status_message") or ""),
          repr(result2.get("status_message","")))

    # ── Test 5: Turn 3 — wrapup trigger ───────────────────────────────────────
    section("5. Turn 3 — wrapup status triggered at turn_number >= 3")
    result3 = await run_turn(pipeline,
        "Furthermore, social media companies profit from outrage and division. "
        "Their algorithms deliberately show controversial content to keep users engaged longer.",
        "turn_3")

    check("turn_number == 3",
          result3.get("turn_number") == 3,
          str(result3.get("turn_number")))
    check("status contains 'Choose an option'",
          "Choose an option" in (result3.get("status_message") or ""),
          repr(result3.get("status_message","")))
    check("argument_scores now has 3 entries",
          len(result3.get("argument_scores") or []) == 3,
          f"scores={result3.get('argument_scores')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Done")
    print("  If all ✅ above: pipeline.py is working correctly.")
    print("  Next step: integrate into app.py by replacing Steps 3-7 in analyze_turn.")
    print()


if __name__ == "__main__":
    # Check API key before running
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌  ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    asyncio.run(main())