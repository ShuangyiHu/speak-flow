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