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
      1. Create a feature branch per module
      2. Commit generated files
      3. Open a PR with the review output as PR description
      4. Auto-merge if reviewer decision is APPROVE
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from crew import SpeakFlowDevTeam
from github_integration import GitHubPRManager  # see github_integration.py

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ── Module Definitions ────────────────────────────────────────────────────────
# Add new modules here as the project grows.

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

[To be expanded — run after turn_analyzer is complete]
""",
    },
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(module_key: str = "turn_analyzer", use_github: bool = True):
    if module_key not in MODULES:
        print(f"Unknown module: {module_key}. Available: {list(MODULES.keys())}")
        sys.exit(1)

    module_def = MODULES[module_key]
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SpeakFlow Dev Team — Building: {module_def['module_name']}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Run the crew ──────────────────────────────────────────────────────────
    result = SpeakFlowDevTeam().crew().kickoff(inputs=module_def)

    print(f"\n{'='*60}")
    print(f"  Dev Team completed: {module_def['module_name']}")
    print(f"  Output files in: {output_dir.absolute()}")
    print(f"{'='*60}\n")

    # ── GitHub PR integration ─────────────────────────────────────────────────
    if use_github and os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO"):
        review_path = output_dir / f"{module_def['module_name']}_review.md"
        review_text = review_path.read_text() if review_path.exists() else ""

        pr_manager = GitHubPRManager(
            token=os.getenv("GITHUB_TOKEN"),
            repo=os.getenv("GITHUB_REPO"),
        )
        pr_manager.create_pr_from_output(
            module_name=module_def["module_name"],
            output_dir=output_dir,
            review_text=review_text,
        )
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
