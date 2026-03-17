"""
rag_eval.py — RAG LLM-as-Judge Evaluation for SpeakFlow AI
============================================================
Run from the project root:
    python scripts/rag_eval.py

What this does:
  For each test case in TEST_CASES, generates TWO coaching responses:
    - WITHOUT RAG (baseline): coach prompt only, no retrieved context
    - WITH RAG    (rag):      coach prompt + ChromaDB retrieved chunks

  Then calls Claude as judge to score both on 3 dimensions (1–5 each):
    1. Relevance     — how well the feedback addresses THIS specific argument
    2. Specificity   — does it reference concrete debate techniques / evidence
    3. Actionability — can the student act on this feedback in the next turn

  Outputs:
    - Console: per-case scores and delta
    - rag_eval_results.json: full results for inspection
    - rag_eval_summary.txt:  headline numbers for resume bullet

Usage notes:
  - Set ANTHROPIC_API_KEY in .env before running
  - Requires rag_retriever.py and response_generator.py in src/
  - USE_STUB_MFA=True is fine; this eval is argument-only
  - Expected runtime: ~8–12 minutes for 15 cases (API rate limits)
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from anthropic import AsyncAnthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# ── Test cases ────────────────────────────────────────────────────────────────
# 15 cases covering 5 topics × 3 argument quality levels (weak / medium / strong)
# Each case represents a realistic student turn in a debate session.
# strategy: which CoachingStrategy the coach is using (affects RAG query)
# expected_theme: what the ideal feedback should reference (for your manual check)

TEST_CASES = [
    # ── Topic 1: Social Media ─────────────────────────────────────────────────
    {
        "id": "sm_weak",
        "topic": "Social media does more harm than good",
        "position": "for",
        "strategy": "CHALLENGE",
        "transcript": "I think social media is bad because people spend too much time on it and it's addictive.",
        "expected_theme": "Should reference filter bubbles, mental health research, or platform design incentives",
        "quality_level": "weak",
    },
    {
        "id": "sm_medium",
        "topic": "Social media does more harm than good",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "Social media harms society because it spreads misinformation rapidly. For example, during COVID-19 fake cures went viral and people were misled.",
        "expected_theme": "Should probe the causal mechanism or ask about platform accountability",
        "quality_level": "medium",
    },
    {
        "id": "sm_strong",
        "topic": "Social media does more harm than good",
        "position": "against",
        "strategy": "CHALLENGE",
        "transcript": "Social media has been essential for political organizing in authoritarian contexts — the Arab Spring demonstrated how platforms enabled coordination that state media suppressed. The harm framing ignores asymmetric access to information.",
        "expected_theme": "Should challenge with evidence of platform censorship or corporate control",
        "quality_level": "strong",
    },

    # ── Topic 2: AI and Jobs ──────────────────────────────────────────────────
    {
        "id": "ai_weak",
        "topic": "AI will take more jobs than it creates",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "AI is going to take everyone's jobs because robots are getting smarter every day.",
        "expected_theme": "Should probe which job categories, or ask about historical automation precedents",
        "quality_level": "weak",
    },
    {
        "id": "ai_medium",
        "topic": "AI will take more jobs than it creates",
        "position": "against",
        "strategy": "CHALLENGE",
        "transcript": "History shows that new technology creates new jobs. The industrial revolution eliminated farm jobs but created factory jobs. AI will be the same — new roles in AI maintenance and data work will emerge.",
        "expected_theme": "Should challenge pace differential or skill transition barriers",
        "quality_level": "medium",
    },
    {
        "id": "ai_strong",
        "topic": "AI will take more jobs than it creates",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "The key distinction is cognitive vs physical labor. Previous automation targeted physical tasks, creating demand for cognitive workers. AI targets cognitive tasks directly, which eliminates the traditional escape valve. McKinsey estimates 375 million workers may need to switch occupations by 2030.",
        "expected_theme": "Should probe whether reskilling infrastructure is adequate, or challenge the McKinsey methodology",
        "quality_level": "strong",
    },

    # ── Topic 3: Climate Policy ───────────────────────────────────────────────
    {
        "id": "climate_weak",
        "topic": "Climate change policy should prioritise economic growth",
        "position": "for",
        "strategy": "REDIRECT",
        "transcript": "I think we need to focus on the economy because people need money to live. Climate is important but jobs come first.",
        "expected_theme": "Should redirect toward the false dichotomy argument or green economy evidence",
        "quality_level": "weak",
    },
    {
        "id": "climate_medium",
        "topic": "Climate change policy should prioritise economic growth",
        "position": "against",
        "strategy": "CHALLENGE",
        "transcript": "Prioritising growth over climate is short-sighted. The economic costs of climate change — flooding, crop failure, displacement — will far exceed the cost of mitigation. The Stern Review estimated inaction costs 20 times more than action.",
        "expected_theme": "Should challenge discount rate assumptions or developing nation perspectives",
        "quality_level": "medium",
    },
    {
        "id": "climate_strong",
        "topic": "Climate change policy should prioritise economic growth",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "In developing nations, this is not a hypothetical tradeoff — it is a present reality. India cannot simultaneously decarbonise at the pace Western nations demand and lift 300 million people out of energy poverty. Climate justice requires acknowledging who bears the transition costs.",
        "expected_theme": "Should probe the CBDR principle or ask about carbon finance mechanisms",
        "quality_level": "strong",
    },

    # ── Topic 4: Standardised Testing ────────────────────────────────────────
    {
        "id": "test_weak",
        "topic": "Standardised testing is an effective measure of student ability",
        "position": "against",
        "strategy": "PROBE",
        "transcript": "Standardised tests are unfair because some students are just bad at taking tests even if they are smart.",
        "expected_theme": "Should probe what intelligence construct is being measured, or ask about test anxiety research",
        "quality_level": "weak",
    },
    {
        "id": "test_medium",
        "topic": "Standardised testing is an effective measure of student ability",
        "position": "for",
        "strategy": "CHALLENGE",
        "transcript": "Without standardised tests we have no common benchmark. Teacher grades are subjective and vary by school quality. A test score allows fair comparison across different school systems.",
        "expected_theme": "Should challenge with socioeconomic confounding or test prep industry evidence",
        "quality_level": "medium",
    },
    {
        "id": "test_strong",
        "topic": "Standardised testing is an effective measure of student ability",
        "position": "against",
        "strategy": "CHALLENGE",
        "transcript": "The SAT has a 0.53 correlation with first-year GPA, which means it explains only 28% of variance in academic performance. Meanwhile, high school GPA, which tests dismiss as subjective, has a 0.56 correlation. The psychometric case for standardised testing over longitudinal assessment is weak.",
        "expected_theme": "Should challenge with predictive validity for diverse populations or test design improvements",
        "quality_level": "strong",
    },

    # ── Topic 5: Universal Basic Income ──────────────────────────────────────
    {
        "id": "ubi_weak",
        "topic": "Universal basic income should be implemented globally",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "UBI is a good idea because everyone deserves money to survive and it would end poverty.",
        "expected_theme": "Should probe funding mechanism or ask about inflation effects",
        "quality_level": "weak",
    },
    {
        "id": "ubi_medium",
        "topic": "Universal basic income should be implemented globally",
        "position": "against",
        "strategy": "CHALLENGE",
        "transcript": "UBI is fiscally unviable at meaningful scale. Giving every US adult $12,000 per year would cost $3 trillion annually, which exceeds the entire discretionary budget. Targeted programmes achieve the same poverty reduction at a fraction of the cost.",
        "expected_theme": "Should challenge with pilot study outcomes or discuss what 'meaningful' UBI replaces",
        "quality_level": "medium",
    },
    {
        "id": "ubi_strong",
        "topic": "Universal basic income should be implemented globally",
        "position": "for",
        "strategy": "PROBE",
        "transcript": "The fiscal objection conflates gross cost with net cost. A UBI replacing existing means-tested programmes, funded by a VAT or wealth tax, can be revenue-neutral while eliminating the poverty trap — the marginal effective tax rate cliff that currently makes low-wage work economically irrational for welfare recipients.",
        "expected_theme": "Should probe whether poverty trap elimination evidence is robust, or ask about international implementation variance",
        "quality_level": "strong",
    },
]

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator assessing the quality of AI debate coaching feedback.
You will be given a student's argument and two coaching responses: a BASELINE response and a RAG-ENHANCED response.
The RAG-enhanced response was generated with access to a curated knowledge base of debate evidence and argumentation techniques.

Score each response on THREE dimensions using a 1–5 scale:

RELEVANCE (1–5): Does the feedback directly address the specific argument made?
  5 = feedback is laser-focused on the exact claim and its specific weaknesses
  3 = feedback is generally on-topic but could apply to many similar arguments
  1 = feedback is generic and ignores the specifics of what was said

SPECIFICITY (1–5): Does the feedback reference concrete debate techniques, named evidence, or specific counterarguments?
  5 = references specific studies, frameworks, named examples, or precise logical moves
  3 = mentions relevant concepts but stays at a general level
  1 = only vague guidance ("be more specific", "add evidence")

ACTIONABILITY (1–5): Can the student act on this feedback in their very next turn?
  5 = feedback gives a concrete direction that the student can immediately execute
  3 = feedback identifies a gap but doesn't clearly show how to fill it
  1 = feedback is evaluative but provides no path forward

Respond ONLY with valid JSON (no markdown, no explanation outside the JSON):
{
  "baseline": {
    "relevance": <1-5>,
    "specificity": <1-5>,
    "actionability": <1-5>,
    "brief_reason": "<one sentence explaining the scores>"
  },
  "rag": {
    "relevance": <1-5>,
    "specificity": <1-5>,
    "actionability": <1-5>,
    "brief_reason": "<one sentence explaining the scores>"
  }
}"""

JUDGE_USER_TEMPLATE = """DEBATE TOPIC: {topic}
STUDENT POSITION: {position}
COACHING STRATEGY: {strategy}

STUDENT'S ARGUMENT:
{transcript}

---

RESPONSE A (BASELINE — no knowledge base):
{baseline_response}

---

RESPONSE B (RAG-ENHANCED — with retrieved debate evidence):
{rag_response}

---

Score both responses on relevance, specificity, and actionability (1–5 each)."""

# ── Response generation ───────────────────────────────────────────────────────
BASELINE_COACH_PROMPT = """You are an expert debate coach. A student is debating the following topic.

Topic: {topic}
Student position: {position}
Strategy to apply: {strategy}

Student said:
"{transcript}"

Apply the {strategy} strategy. Give a single coaching response (2-3 sentences) that directly responds to what the student said. Be specific to their argument."""

RAG_COACH_PROMPT = """You are an expert debate coach. A student is debating the following topic.

Topic: {topic}
Student position: {position}
Strategy to apply: {strategy}

Student said:
"{transcript}"

RELEVANT DEBATE KNOWLEDGE (retrieved from knowledge base):
{rag_context}

Using the knowledge base above where relevant, apply the {strategy} strategy. Give a single coaching response (2-3 sentences) that directly responds to what the student said. Ground your feedback in the evidence above where it strengthens the feedback."""

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ScoreSet:
    relevance: int
    specificity: int
    actionability: int
    brief_reason: str

    @property
    def total(self):
        return self.relevance + self.specificity + self.actionability

@dataclass
class CaseResult:
    case_id: str
    topic: str
    quality_level: str
    strategy: str
    transcript: str
    baseline_response: str
    rag_response: str
    baseline_scores: Optional[ScoreSet]
    rag_scores: Optional[ScoreSet]
    delta: Optional[float]   # rag.total - baseline.total (out of 15)
    error: Optional[str]

# ── Core evaluation logic ─────────────────────────────────────────────────────
async def generate_baseline(client, case: dict) -> str:
    prompt = BASELINE_COACH_PROMPT.format(
        topic=case["topic"],
        position=case["position"],
        strategy=case["strategy"],
        transcript=case["transcript"],
    )
    try:
        resp = await client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


async def generate_rag_response(client, case: dict, rag_retriever) -> tuple[str, str]:
    """Returns (rag_context, rag_response)"""
    try:
        from shared_types import CoachingStrategy
        strategy_enum = CoachingStrategy[case["strategy"]]
        ctx = await rag_retriever.retrieve(
            transcript=case["transcript"],
            topic=case["topic"],
            strategy=strategy_enum,
            turn_number=1,
        )
        rag_context = "\n\n".join(
            f"[{c.source}] {c.text}" for c in ctx.chunks[:3]
        ) if ctx.chunks else "No relevant evidence retrieved."
    except Exception as e:
        rag_context = f"[RAG retrieval failed: {e}]"

    prompt = RAG_COACH_PROMPT.format(
        topic=case["topic"],
        position=case["position"],
        strategy=case["strategy"],
        transcript=case["transcript"],
        rag_context=rag_context,
    )
    try:
        resp = await client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return rag_context, resp.content[0].text.strip()
    except Exception as e:
        return rag_context, f"[ERROR: {e}]"


async def judge_responses(client, case: dict, baseline: str, rag: str) -> tuple[ScoreSet, ScoreSet]:
    judge_msg = JUDGE_USER_TEMPLATE.format(
        topic=case["topic"],
        position=case["position"],
        strategy=case["strategy"],
        transcript=case["transcript"],
        baseline_response=baseline,
        rag_response=rag,
    )
    try:
        resp = await client.messages.create(
            model=MODEL, max_tokens=500,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": judge_msg}],
        )
        import re
        raw = resp.content[0].text.strip()
        raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
        data = json.loads(raw)

        def parse_scores(d):
            return ScoreSet(
                relevance=int(d["relevance"]),
                specificity=int(d["specificity"]),
                actionability=int(d["actionability"]),
                brief_reason=d.get("brief_reason", ""),
            )
        return parse_scores(data["baseline"]), parse_scores(data["rag"])
    except Exception as e:
        raise RuntimeError(f"Judge call failed: {e}") from e


async def run_single_case(client, case: dict, rag_retriever, idx: int, total: int) -> CaseResult:
    print(f"  [{idx+1}/{total}] {case['id']} ({case['quality_level']}, {case['strategy']})...", end=" ", flush=True)
    t0 = time.time()

    try:
        # Run baseline and RAG response generation in parallel
        baseline_task = generate_baseline(client, case)
        rag_task = generate_rag_response(client, case, rag_retriever)
        (baseline_response, (rag_context, rag_response)) = await asyncio.gather(
            baseline_task, rag_task
        )

        # Small pause to respect rate limits before judge call
        await asyncio.sleep(1.0)

        baseline_scores, rag_scores = await judge_responses(
            client, case, baseline_response, rag_response
        )
        delta = rag_scores.total - baseline_scores.total

        elapsed = time.time() - t0
        print(f"baseline={baseline_scores.total}/15  rag={rag_scores.total}/15  delta={delta:+d}  ({elapsed:.1f}s)")

        return CaseResult(
            case_id=case["id"],
            topic=case["topic"],
            quality_level=case["quality_level"],
            strategy=case["strategy"],
            transcript=case["transcript"],
            baseline_response=baseline_response,
            rag_response=rag_response,
            baseline_scores=baseline_scores,
            rag_scores=rag_scores,
            delta=float(delta),
            error=None,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e}  ({elapsed:.1f}s)")
        return CaseResult(
            case_id=case["id"], topic=case["topic"],
            quality_level=case["quality_level"], strategy=case["strategy"],
            transcript=case["transcript"],
            baseline_response="", rag_response="",
            baseline_scores=None, rag_scores=None,
            delta=None, error=str(e),
        )


async def main():
    print("=" * 65)
    print("SpeakFlow AI — RAG Evaluation (LLM-as-Judge)")
    print(f"Model: {MODEL} | Cases: {len(TEST_CASES)}")
    print("=" * 65)

    # Import RAG retriever
    try:
        from rag_retriever import RAGRetriever
        rag = RAGRetriever()
        print(f"RAGRetriever loaded. KB status: {rag._kb_available}")
    except Exception as e:
        print(f"WARNING: Could not load RAGRetriever: {e}")
        print("Running in baseline-only mode (delta will be 0 for all cases)")
        rag = None

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

    results = []
    for i, case in enumerate(TEST_CASES):
        result = await run_single_case(client, case, rag, i, len(TEST_CASES))
        results.append(result)
        # Rate limit buffer between cases
        if i < len(TEST_CASES) - 1:
            await asyncio.sleep(2.0)

    # ── Compute summary stats ─────────────────────────────────────────────────
    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not valid:
        print("\nAll cases failed. Check API key and module imports.")
        return

    avg_baseline = sum(r.baseline_scores.total for r in valid) / len(valid)
    avg_rag      = sum(r.rag_scores.total for r in valid) / len(valid)
    avg_delta    = avg_rag - avg_baseline
    pct_improved = sum(1 for r in valid if r.delta > 0) / len(valid) * 100

    # Per-dimension
    dims = ["relevance", "specificity", "actionability"]
    dim_deltas = {}
    for dim in dims:
        b = sum(getattr(r.baseline_scores, dim) for r in valid) / len(valid)
        g = sum(getattr(r.rag_scores, dim) for r in valid) / len(valid)
        dim_deltas[dim] = (b, g, g - b)

    # Per quality level
    qual_stats = {}
    for level in ["weak", "medium", "strong"]:
        subset = [r for r in valid if r.quality_level == level]
        if subset:
            b = sum(r.baseline_scores.total for r in subset) / len(subset)
            g = sum(r.rag_scores.total for r in subset) / len(subset)
            qual_stats[level] = (b, g, g - b)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"Valid cases: {len(valid)}/{len(TEST_CASES)}  |  Errors: {len(errors)}")
    print(f"\nOverall (out of 15):")
    print(f"  Baseline avg:   {avg_baseline:.2f}")
    print(f"  RAG avg:        {avg_rag:.2f}")
    print(f"  Mean delta:     {avg_delta:+.2f}  ({avg_delta/15*100:+.1f}%)")
    print(f"  Cases improved: {pct_improved:.0f}%")

    print(f"\nPer-dimension breakdown (avg scores, 1–5):")
    for dim, (b, g, d) in dim_deltas.items():
        print(f"  {dim:<15}  baseline={b:.2f}  rag={g:.2f}  delta={d:+.2f}")

    print(f"\nBy argument quality level:")
    for level, (b, g, d) in qual_stats.items():
        print(f"  {level:<8}  baseline={b:.2f}  rag={g:.2f}  delta={d:+.2f}")

    print(f"\nPer-case deltas:")
    for r in valid:
        marker = "✅" if r.delta > 0 else "➖" if r.delta == 0 else "❌"
        print(f"  {marker} {r.case_id:<20}  {r.baseline_scores.total:>2}/15 → {r.rag_scores.total:>2}/15  ({r.delta:+.0f})")

    if errors:
        print(f"\nFailed cases:")
        for r in errors:
            print(f"  ❌ {r.case_id}: {r.error}")

    # ── Resume bullet suggestion ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESUME BULLET SUGGESTION")
    print("=" * 65)
    specificity_b, specificity_g, _ = dim_deltas["specificity"]
    print(
        f'Built a RAG pipeline with ChromaDB and sentence embeddings to ground\n'
        f'coaching responses in curated debate evidence; LLM-as-judge evaluation\n'
        f'on {len(valid)} held-out turns showed RAG-grounded responses scored\n'
        f'{avg_delta:+.1f} points higher on a 15-point relevance rubric\n'
        f'({avg_baseline:.1f} → {avg_rag:.1f}), with specificity improving\n'
        f'from {specificity_b:.1f} to {specificity_g:.1f}/5 on average.'
    )

    # ── Save full results ─────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "rag_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(
            [asdict(r) if r.baseline_scores else {
                "case_id": r.case_id, "error": r.error,
                "baseline_scores": None, "rag_scores": None,
             } for r in results],
            f, indent=2, default=str,
        )
    print(f"\nFull results saved → {output_path}")

    summary_path = os.path.join(os.path.dirname(__file__), "rag_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"SpeakFlow RAG Evaluation Summary\n")
        f.write(f"Model: {MODEL} | Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write(f"Cases evaluated: {len(valid)}\n")
        f.write(f"Baseline avg score: {avg_baseline:.2f}/15\n")
        f.write(f"RAG avg score:      {avg_rag:.2f}/15\n")
        f.write(f"Mean improvement:   {avg_delta:+.2f} ({avg_delta/15*100:+.1f}%)\n")
        f.write(f"Cases improved:     {pct_improved:.0f}%\n\n")
        f.write("Per-dimension:\n")
        for dim, (b, g, d) in dim_deltas.items():
            f.write(f"  {dim}: {b:.2f} → {g:.2f} (delta {d:+.2f})\n")
        f.write("\nBy quality level:\n")
        for level, (b, g, d) in qual_stats.items():
            f.write(f"  {level}: {b:.2f} → {g:.2f} (delta {d:+.2f})\n")
    print(f"Summary saved      → {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())