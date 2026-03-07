#!/usr/bin/env python
"""
SpeakFlow AI — Dev Team Runner

Usage:
    python main.py                          # Full pipeline for Turn Analyzer
    python main.py --module coach_policy    # Run for a different module
    python main.py --skip-github            # Run crew only, no GitHub PR
    python main.py --github-only            # Skip crew, push existing output/ to GitHub

Dev loop:
    design → code → review ──APPROVE──→ test → frontend → GitHub PR
                        │
                   REQUEST_CHANGES
                        │
                     revise → re-review → (repeat up to MAX_REVISIONS)
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from crew import SpeakFlowDevTeam
from github_integration import GitHubPRManager

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

MAX_REVISIONS = 3  # Max fix-review cycles before giving up and opening PR anyway

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
- Use Anthropic Python SDK with claude-sonnet-4-20250514

Acceptance criteria:
- analyze() returns TurnAnalysis within 3 seconds for 30-word input
- has_claim=True when transcript contains a clear position statement
- has_claim=False when transcript is a question or off-topic
- argument_score > 0.7 for well-structured CRE argument
- argument_score < 0.4 for opinion-only turn with no reasoning
- Empty transcript returns default TurnAnalysis, no exception raised
- mispronounced_words is empty list when USE_STUB_MFA=True
""",
    },
    "coach_policy": {
        "module_name": "coach_policy.py",
        "class_name": "CoachPolicyAgent",
        "requirements": """
The CoachPolicyAgent is a self-contained Python module for SpeakFlow AI, an English
debate coaching platform for Chinese L2 learners. It receives a TurnAnalysis object
(produced by TurnAnalyzer) and decides the next coaching action: what strategy to apply
and what to say to the student.

The response must feel like a debate PARTNER, not a teacher. Never say "Good job!" or
"You need to improve X." Instead, engage with the argument itself and ask follow-up
questions or push back naturally.

--- DATA MODELS ---

CoachingStrategy (Enum):
  PROBE               - Ask a follow-up question to draw out more reasoning
  CHALLENGE           - Push back on the claim with a counterargument
  REDIRECT            - Steer the student back on topic if they went off track
  PRAISE_AND_PUSH     - Acknowledge a strong point then raise the bar
  CORRECT_PRONUNCIATION - Gently model correct pronunciation mid-conversation

CoachingAction (dataclass):
  strategy: CoachingStrategy
  response_text: str        # What the AI debate partner says out loud
  internal_reason: str      # Why this strategy was chosen (for logging/tracing)
  target_word: str          # Populated only for CORRECT_PRONUNCIATION, else ""
  difficulty_delta: int     # -1 (easier), 0 (same), +1 (harder) — for adaptive difficulty

SessionContext (dataclass):
  session_id: str
  topic: str
  user_position: str                    # "for" or "against"
  turn_number: int
  coaching_history: list[CoachingStrategy]   # strategies used in prior turns
  argument_scores: list[float]               # argument_score from each prior turn

--- MAIN CLASS ---

class CoachPolicyAgent:

  def __init__(self, anthropic_api_key: str)
    - Initialize AsyncAnthropic client
    - Use model: claude-sonnet-4-5
    - LLM call timeout: 30 seconds

  async def decide(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingAction
    - Main entry point
    - Select strategy using _select_strategy()
    - Generate response_text using Claude API via _generate_response()
    - Return CoachingAction
    - Handle empty transcript: return PROBE strategy with an opening question
    - Never raise exceptions — return a default CoachingAction on any error

  def _select_strategy(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingStrategy
    - SYNCHRONOUS method (no LLM needed, pure logic)
    - Rules in priority order:
      1. If analysis.pronunciation.mispronounced_words has severity==MAJOR: CORRECT_PRONUNCIATION
      2. If analysis.turn_input.transcript is empty or off-topic (argument_score < 0.1): REDIRECT
      3. If last 2 entries in coaching_history are the same strategy: force a different one
      4. If argument_score >= 0.7 and turn_number > 2: CHALLENGE
      5. If argument_score >= 0.7: PRAISE_AND_PUSH
      6. If has_claim=True but has_reasoning=False: PROBE
      7. If has_reasoning=True but has_evidence=False: PROBE
      8. Default: PROBE

  async def _generate_response(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str
    - Call Claude API with a prompt that includes:
      - The student's transcript
      - The selected strategy
      - The debate topic and position
      - Last 3 turns of coaching history (to avoid repetition)
    - Response must be 1-3 sentences, conversational, no bullet points
    - For CORRECT_PRONUNCIATION: model the word correctly in a natural sentence
    - For CHALLENGE: take the opposing side clearly but not aggressively
    - Strip any markdown from the response before returning

  def _build_prompt(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str
    - Build the full prompt string for Claude
    - Include strategy-specific instructions
    - Specify: respond in 1-3 sentences, no bullet points, no "Great job!" openers

  def _create_default_action(self, context: SessionContext) -> CoachingAction
    - Return a safe fallback CoachingAction with strategy=PROBE
    - response_text should be a generic opening question relevant to the topic

--- KEY BEHAVIORS ---

- decide() must complete within 10 seconds total
- _select_strategy() must be deterministic given the same inputs (no randomness)
- coaching_history anti-repeat: if the same strategy was used in the last 2 turns,
  skip it and pick the next applicable strategy from the priority list
- difficulty_delta logic:
    +1 if argument_score >= 0.7 for last 2 turns
    -1 if argument_score < 0.3 for last 2 turns
     0 otherwise
- Use Anthropic Python SDK (anthropic package), AsyncAnthropic client
- All Claude responses must have markdown stripped before returning
- Module must be fully self-contained and importable with no missing dependencies

--- ACCEPTANCE CRITERIA ---

- decide() returns CoachingAction within 10 seconds
- _select_strategy() returns PROBE when has_claim=True, has_reasoning=False
- _select_strategy() returns CHALLENGE when argument_score >= 0.7 and turn_number > 2
- _select_strategy() never repeats the same strategy 3 turns in a row
- _select_strategy() returns REDIRECT when argument_score < 0.1
- response_text is 1-3 sentences and contains no markdown
- decide() returns default CoachingAction on empty transcript, no exception raised
- difficulty_delta = +1 when last 2 argument_scores are both >= 0.7
""",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_decision(review_text: str) -> str:
    if not review_text:
        return "REQUEST_CHANGES"
    match = re.search(r"DECISION:\s*(APPROVE|REQUEST_CHANGES)", review_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if "APPROVE" in review_text.upper() and "REQUEST_CHANGES" not in review_text.upper():
        return "APPROVE"
    return "REQUEST_CHANGES"


def _read_output(output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _push_to_github(module_def: dict, output_dir: Path):
    if not (os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO")):
        print("  [GitHub] Skipped — set GITHUB_TOKEN and GITHUB_REPO to enable.")
        return
    review_text = _read_output(output_dir, f"{module_def['module_name']}_review.md")
    pr_manager = GitHubPRManager(
        token=os.getenv("GITHUB_TOKEN"),
        repo=os.getenv("GITHUB_REPO"),
    )
    pr_manager.create_pr_from_output(
        module_name=module_def["module_name"],
        output_dir=output_dir,
        review_text=review_text,
    )


# ── Runners ───────────────────────────────────────────────────────────────────

def run(module_key: str = "turn_analyzer", use_github: bool = True):
    module_def = MODULES[module_key]
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    module_name = module_def["module_name"]

    print(f"\n{'='*60}")
    print(f"  SpeakFlow Dev Team — Building: {module_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Step 1: Main pipeline (design → code → review) ────────────────────────
    SpeakFlowDevTeam().crew().kickoff(inputs=module_def)

    review_text = _read_output(output_dir, f"{module_name}_review.md")
    decision = _extract_decision(review_text)

    print(f"\n  [Review] Initial decision: {decision}")

    # ── Step 2: Revision loop ─────────────────────────────────────────────────
    revision = 0
    while decision == "REQUEST_CHANGES" and revision < MAX_REVISIONS:
        revision += 1
        print(f"\n{'='*60}")
        print(f"  [Revision {revision}/{MAX_REVISIONS}] Fixing issues in {module_name}...")
        print(f"{'='*60}\n")

        current_code = _read_output(output_dir, module_name)

        SpeakFlowDevTeam().revision_crew(
            module_name=module_name,
            review_feedback=review_text,
            current_code=current_code,
        ).kickoff()

        review_text = _read_output(output_dir, f"{module_name}_review.md")
        decision = _extract_decision(review_text)
        print(f"\n  [Review] Revision {revision} decision: {decision}")

    # ── Step 3: Report outcome ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if decision == "APPROVE":
        print(f"  ✅ APPROVED after {revision} revision(s) — running test + frontend tasks")
    else:
        print(f"  ⚠️  Still REQUEST_CHANGES after {MAX_REVISIONS} revision(s)")
        print(f"     Opening PR anyway so you can review manually on GitHub")
    print(f"{'='*60}\n")

    # ── Step 4: Test + frontend (run regardless — let CI catch failures) ───────
    # These are separate crews so we don't re-run design/code
    test_frontend_inputs = {**module_def}
    dev_team = SpeakFlowDevTeam()

    from crewai import Crew, Process
    post_crew = Crew(
        agents=[dev_team.test_engineer(), dev_team.frontend_engineer()],
        tasks=[dev_team.test_task(), dev_team.frontend_task()],
        process=Process.sequential,
        verbose=True,
    )
    post_crew.kickoff(inputs=test_frontend_inputs)

    # ── Step 5: GitHub PR ─────────────────────────────────────────────────────
    if use_github:
        _push_to_github(module_def, output_dir)


def run_github(module_key: str):
    """Push existing output/ files to GitHub PR without re-running crew."""
    module_def = MODULES[module_key]
    output_dir = Path("output")
    if not output_dir.exists():
        print("  [GitHub] No output/ directory found. Run without --github-only first.")
        sys.exit(1)
    print(f"\n  [GitHub only] Pushing {module_def['module_name']} to GitHub...")
    _push_to_github(module_def, output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeakFlow Dev Team Runner")
    parser.add_argument("--module", default="turn_analyzer", choices=list(MODULES.keys()))
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub PR")
    parser.add_argument("--github-only", action="store_true", help="Push existing output to GitHub only")
    args = parser.parse_args()

    if args.module not in MODULES:
        print(f"Unknown module. Available: {list(MODULES.keys())}")
        sys.exit(1)

    if args.github_only:
        run_github(module_key=args.module)
    else:
        run(module_key=args.module, use_github=not args.skip_github)