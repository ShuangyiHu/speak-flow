"""
app.py — SpeakFlow AI Gradio frontend
======================================
⚠️  UI DESIGN IS FROZEN — DO NOT MODIFY LAYOUT WITHOUT EXPLICIT APPROVAL ⚠️

Approved layout (v2, locked):
  - Header: 🎙 SpeakFlow AI — Debate Coach  (dark title, subtitle)
  - Row 1: Debate Topic dropdown (left, wide) + Your Position radio For/Against (right)
  - Row 2 (two columns, 1:1):
      LEFT  — Record Audio component, "Or type transcript here" textbox,
               [Analyze Turn] primary btn + [New Session] secondary btn
      RIGHT — Transcript textbox, Coach Response textbox, Difficulty label
  - Row 3 (Status bar — full width)
  - Row 4 (Argument Analysis): Has Claim / Has Reasoning / Has Evidence checkboxes,
           Argument Score (0-1) number, Summary textbox
  - Row 5 (Pronunciation Feedback): Mispronounced Words textbox

Theme: gr.themes.Soft()  — do not change theme
Accent colour: indigo/blue (#4F46E5 family) — set via theme, not inline CSS

Debate topics (locked list — add only, never remove):
  1. Social media does more harm than good
  2. AI will take more jobs than it creates
  3. Climate change policy should prioritise economic growth
  4. Standardised testing is an effective measure of student ability
  5. Universal basic income should be implemented globally
  6. Space exploration funding should be redirected to solving poverty
  7. Renewable energy can fully replace fossil fuels by 2050

Analysis pipeline (real, not stub):
  audio_file  →  Whisper (openai-whisper)  →  transcript
  transcript  →  TurnAnalyzer.analyze()    →  TurnAnalysis
  TurnAnalysis → CoachPolicyAgent.decide() →  CoachingAction
  CoachingAction → ResponseGenerator       →  GeneratedResponse

Env vars required:
  ANTHROPIC_API_KEY   — Anthropic Claude API key
  USE_STUB_MFA=True   — skip MFA (default for local dev)

Optional:
  ANTHROPIC_MODEL     — defaults to claude-sonnet-4-5
  WHISPER_MODEL       — tiny / base / small / medium / large  (default: base)

If you need to change layout, create a new file app_v3.py and get approval before
replacing this file.
"""

import asyncio
import os
import sys
import time
import logging

import gradio as gr
from dotenv import load_dotenv

# Load .env from project root (two levels up from app/app.py)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from turn_analyzer import TurnAnalyzer
from coach_policy import CoachPolicyAgent
from response_generator import ResponseGenerator
from shared_types import (
    TurnInput,
    CoachingStrategy,
    SessionContext,
    TurnIntent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
os.environ.setdefault("USE_STUB_MFA", "True")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Whisper lazy-load ─────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper_model():
    """Lazy-load Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            model_name = os.getenv("WHISPER_MODEL", "base")
            logger.info(f"Loading Whisper model: {model_name}")
            _whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.error("openai-whisper not installed. Run: pip install openai-whisper")
            _whisper_model = None
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper. Returns empty string on failure."""
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""


# ── Frozen debate topic list ──────────────────────────────────────────────────
DEBATE_TOPICS = [
    "Social media does more harm than good",
    "AI will take more jobs than it creates",
    "Climate change policy should prioritise economic growth",
    "Standardised testing is an effective measure of student ability",
    "Universal basic income should be implemented globally",
    "Space exploration funding should be redirected to solving poverty",
    "Renewable energy can fully replace fossil fuels by 2050",
]


# ── State class ───────────────────────────────────────────────────────────────
class SpeakFlowUI:
    def __init__(self):
        self.turn_analyzer     = TurnAnalyzer(anthropic_api_key=ANTHROPIC_API_KEY)
        self.coach             = CoachPolicyAgent(mfa_enabled=False)
        self.response_gen      = ResponseGenerator()

        self.turn_count: int         = 0
        self.prior_turns: list[str]  = []
        self.prior_responses: list[str] = []
        self.argument_scores: list[float] = []
        self.coaching_history: list[CoachingStrategy] = []
        self.last_coach_question: str = ""
        self.last_turn_intent: str = ""
        self.session_transcripts: list[str] = []   # full transcripts for end summary
        self.session_language_tips: list[str] = []  # per-turn language tips for end summary
        self.session_topic: str = ""
        self.session_position: str = ""
        self.in_wrapup: bool = False  # block further analysis during wrapup

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset_session(self):
        self.turn_count = 0
        self.prior_turns = []
        self.prior_responses = []
        self.argument_scores = []
        self.coaching_history = []
        self.last_coach_question = ""
        self.last_turn_intent = ""
        self.session_transcripts = []
        self.session_language_tips = []
        self.session_topic = ""
        self.session_position = ""
        self.in_wrapup = False
        return (
            None,
            "", "", "", 
            "🔄 Session reset. Record your first argument.",
            False, False, False, False,
            "", "",
            gr.update(visible=False),
            gr.update(visible=False), "", "",  # session_summary_row, debate_summary, lang_summary
            gr.update(interactive=True),  # re-enable analyze btn
        )

    def continue_session(self):
        """Continue debating — hide wrap-up buttons, reset turn counter, keep history."""
        self.turn_count = 2   # will become 3 on next turn increment, keeping probing
        self.in_wrapup = False
        return (
            None,
            "", "", "",
            "▶ Continuing — record your next argument above.",
            False, False, False, False,
            "", "",
            gr.update(visible=False),
            gr.update(visible=False), "", "",
            gr.update(interactive=True),
        )

    async def stop_session(self):
        """Generate end-of-session summaries for both debate and language."""
        self.in_wrapup = True

        debate_summary = "Great session! You built up your argument well across the turns."
        lang_summary = "Keep practising connectors like 'therefore' and 'however' to link your ideas."

        if self.session_transcripts:
            try:
                debate_summary = await self.response_gen.generate_session_debate_summary(
                    topic=self.session_topic,
                    user_position=self.session_position,
                    transcripts=self.session_transcripts,
                    argument_scores=self.argument_scores,
                )
            except Exception as e:
                logger.error(f"Debate summary failed: {e}")

            try:
                lang_summary = await self.response_gen.generate_session_language_summary(
                    transcripts=self.session_transcripts,
                    per_turn_tips=self.session_language_tips,
                )
            except Exception as e:
                logger.error(f"Language summary failed: {e}")

        # Only update the 4 components wired to stop_btn
        return (
            gr.update(visible=False),   # hide wrapup_row
            gr.update(visible=True),    # show session_summary_row
            debate_summary,             # debate_summary_out
            lang_summary,               # lang_summary_out
        )

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze_turn(self, audio_file, typed_transcript, topic, position):
        """
        Full pipeline:
          1. Transcribe audio (Whisper) — or use typed transcript
          2. TurnAnalyzer.analyze()     — argument + pronunciation
          3. CoachPolicyAgent.decide()  — strategy selection
          4. ResponseGenerator          — natural language response
        """
        t_start = time.time()

        # Block analysis during wrapup — user should click a button
        if self.in_wrapup:
            return (
                None, "", "", "",
                "⏸ Round complete — please choose Continue, New Topic, or Stop below.",
                False, False, False, False, "", "",
                gr.update(visible=True),
                gr.update(visible=False), "", "",
                gr.update(interactive=False),
            )

        # Remember topic/position for end summary
        self.session_topic = topic or DEBATE_TOPICS[0]
        self.session_position = (position or "For").lower()

        # ── Step 1: Get transcript ────────────────────────────────────────────
        if typed_transcript and typed_transcript.strip():
            transcript = typed_transcript.strip()
            transcript_source = "typed"
        elif audio_file is not None:
            transcript = transcribe_audio(audio_file)
            transcript_source = "whisper"
            if not transcript:
                return (
                    None, "", "⚠️ Transcription failed — check Whisper installation.", "",
                    "❌ Transcription error.", False, False, False, False, "", "",
                    gr.update(visible=False),
                    gr.update(visible=False), "", "", gr.update(interactive=True),
                )
        else:
            return (
                None, "", "Please record audio or type a transcript first.", "",
                "⚠️ No input provided.", False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        self.turn_count += 1

        # ── Step 2: Intent detection — route before scoring ───────────────────
        intent = self.turn_analyzer._detect_intent(transcript)
        self.last_turn_intent = intent.value

        # ── META_QUESTION path: student is asking the coach something ─────────
        if intent == TurnIntent.META_QUESTION:
            meta_response = self._handle_meta_question(transcript, topic, position)
            elapsed = time.time() - t_start
            status = f"💬 Turn {self.turn_count} — Coach answered your question. Continue debating above."
            return (
                None, transcript, meta_response, "", status,
                False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        # ── Step 3: TurnAnalyzer (only for DEBATE_STATEMENT) ─────────────────
        turn_input = TurnInput(
            transcript=transcript,
            session_id="gradio-session",
            turn_number=self.turn_count,
            topic=topic or DEBATE_TOPICS[0],
            user_position=(position or "For").lower(),
            audio_path=audio_file or "",
            prior_turns=[{"summary": t} for t in self.prior_turns[-3:]],
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(
                self.turn_analyzer.analyze(turn_input)
            )
        except Exception as e:
            logger.error(f"TurnAnalyzer failed: {e}")
            return (
                None, transcript,
                f"⚠️ Analysis failed: {e}", "",
                f"❌ Turn {self.turn_count} — analysis error.",
                False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        self.prior_turns.append(analysis.argument.summary or transcript)
        self.argument_scores.append(analysis.argument.argument_score)
        self.session_transcripts.append(transcript)  # store full transcript for end summary

        # ── Step 4: CoachPolicyAgent ──────────────────────────────────────────
        session_ctx = SessionContext(
            session_id="gradio-session",
            topic=topic or DEBATE_TOPICS[0],
            user_position=(position or "For").lower(),
            turn_number=self.turn_count,
            coaching_history=list(self.coaching_history),
            argument_scores=list(self.argument_scores),
            last_coach_question=self.last_coach_question,
            last_turn_intent=self.last_turn_intent,
        )

        try:
            coaching_action = loop.run_until_complete(
                self.coach.decide(analysis, session_ctx)
            )
            # Build rich intent context for the response generator
            arg = analysis.argument
            dim_lines = []
            for label, val, threshold in [
                ("Clarity",   arg.clarity_score,     0.5),
                ("Reasoning", arg.reasoning_score,   0.5),
                ("Depth",     arg.depth_score,        0.4),
                ("Fluency",   arg.fluency_score_arg,  0.5),
            ]:
                status = "✓ good" if val >= threshold else "✗ weak"
                dim_lines.append(f"  {label}: {val:.1f} {status}")

            missing = [
                ("clearer position statement", arg.clarity_score < 0.5),
                ("logical reasoning explaining WHY", arg.reasoning_score < 0.5),
                ("a concrete example or elaboration", arg.depth_score < 0.4),
                ("cleaner grammar and connectives", arg.fluency_score_arg < 0.5),
            ]
            weakest = [label for label, is_weak in missing if is_weak]

            coaching_action.intent = "\n".join([
                f"Topic: {topic or DEBATE_TOPICS[0]}",
                f"Student position: {(position or 'For').lower()}",
                f"Turn {self.turn_count}. Student just said:",
                f'  "{transcript[:300]}"',
                f"Scores:",
                *dim_lines,
                f"Coach summary: {arg.summary}",
                f"Weakest areas this turn: {', '.join(weakest) if weakest else 'none — argument is strong'}",
                f"Coach last asked: \"{self.last_coach_question}\"",
                f"Do NOT repeat or rephrase the coach's last question.",
            ])
        except Exception as e:
            logger.error(f"CoachPolicyAgent failed: {e}")
            coaching_action = None

        if coaching_action:
            self.coaching_history.append(coaching_action.strategy)

        # ── Step 5: ResponseGenerator ─────────────────────────────────────────
        coach_text = ""
        improved_text = ""
        language_tips = ""
        if coaching_action:
            from shared_types import ResponseRequest
            req = ResponseRequest(
                coaching_action=coaching_action,
                topic=topic or DEBATE_TOPICS[0],
                user_position=(position or "For").lower(),
                prior_responses=list(self.prior_responses),
                turn_number=self.turn_count,
            )
            try:
                response = loop.run_until_complete(
                    self.response_gen.generate_response(req)
                )
                strategy_label = coaching_action.strategy.value.title()
                coach_text = f"[{strategy_label}]\n\n{response.text}"
                self.last_coach_question = response.text
                self.prior_responses.append(response.text)
                if len(self.prior_responses) > 5:
                    self.prior_responses.pop(0)
            except Exception as e:
                logger.error(f"ResponseGenerator failed: {e}")
                coach_text = f"[{coaching_action.strategy.value.title()}]\n\n(Response unavailable: {e})"

            # Improved version — "How to say it better" box
            try:
                improved_text = loop.run_until_complete(
                    self.response_gen.generate_improved_version(
                        original_transcript=transcript,
                        topic=topic or DEBATE_TOPICS[0],
                        user_position=(position or "For").lower(),
                        vocabulary_flags=analysis.argument.vocabulary_flags,
                    )
                )
            except Exception as e:
                logger.error(f"Improved version generation failed: {e}")
                improved_text = ""

            # Language Tips — grammar/vocabulary/sentence structure only, NOT content or logic
            try:
                language_tips = loop.run_until_complete(
                    self.response_gen.generate_language_tips(
                        original_transcript=transcript,
                        improved_transcript=improved_text,
                        vocabulary_flags=analysis.argument.vocabulary_flags,
                        clarity_feedback=analysis.argument.clarity_feedback or "",
                        fluency_feedback=analysis.argument.fluency_feedback or "",
                    )
                )
            except Exception as e:
                logger.error(f"Language tips generation failed: {e}")
                language_tips = ""

        loop.close()

        # Store per-turn language tip for end summary
        self.session_language_tips.append(language_tips)

        # Set wrapup state so further analysis is blocked
        show_wrapup = self.turn_count >= 3
        if show_wrapup:
            self.in_wrapup = True

        # ── Pronunciation display ─────────────────────────────────────────────
        errors = analysis.pronunciation.mispronounced_words
        if errors:
            lines = []
            for w in errors:
                lines.append(
                    f"• {w.word}  expected: {w.expected_ipa}  actual: {w.actual_ipa}"
                    f"  [{w.severity.value}]"
                )
            pronunciation_text = "\n".join(lines)
        else:
            pronunciation_text = "✓ No pronunciation issues detected"

        # ── Score display ─────────────────────────────────────────────────────
        arg = analysis.argument
        score = arg.argument_score
        score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        elapsed = time.time() - t_start
        status = (
            f"✅ Turn {self.turn_count} complete  "
            f"{score_emoji} Score: {score:.2f}  "
            f"⏱ {elapsed:.1f}s ({transcript_source})"
            + (" — Choose an option below" if show_wrapup else " — Record your next argument above")
        )

        return (
            None,
            transcript,
            coach_text,
            improved_text,
            status,
            gr.update(value=arg.clarity_score >= 0.5,     label=f"Clarity  {arg.clarity_score:.1f}"),
            gr.update(value=arg.reasoning_score >= 0.5,   label=f"Reasoning  {arg.reasoning_score:.1f}"),
            gr.update(value=arg.depth_score >= 0.4,       label=f"Depth  {arg.depth_score:.1f}"),
            gr.update(value=arg.fluency_score_arg >= 0.5, label=f"Fluency  {arg.fluency_score_arg:.1f}"),
            language_tips,
            pronunciation_text,
            gr.update(visible=show_wrapup),
            gr.update(visible=False), "", "",   # session summary hidden during debate
            gr.update(interactive=not show_wrapup),  # disable analyze when in wrapup
        )

    # ── Meta-question handler ─────────────────────────────────────────────────
    def _handle_meta_question(self, transcript: str, topic: str, position: str) -> str:
        """
        Student is pushing back or asking the coach to clarify (e.g. "Didn't I just give you a reason?").
        Coach must:
        1. Acknowledge what the student said was correct / valid
        2. Clarify or rephrase — NOT repeat the same question
        3. Move the conversation forward to the NEXT weakest area
        """
        from shared_types import ResponseRequest, CoachingAction, CoachingStrategy
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            dummy_action = CoachingAction(
                strategy=CoachingStrategy.PROBE,
                intent=(
                    f"The coach previously asked: \"{self.last_coach_question}\"\n"
                    f"The student pushed back by saying: \"{transcript}\"\n"
                    f"The student is RIGHT — they DID answer the previous question.\n"
                    f"You must: (1) briefly acknowledge they answered it, "
                    f"(2) NOT repeat or rephrase the same question, "
                    f"(3) ask ONE new question that moves the debate forward — "
                    f"either ask them to go deeper on what they said, or address a different angle of the topic."
                ),
                target_claim=None,
                target_word=None,
                target_phoneme=None,
                argument_score=0.0,
                pronunciation_score=1.0,
                difficulty_delta=0,
                turn_number=self.turn_count,
                topic=topic or DEBATE_TOPICS[0],
                user_position=(position or "For").lower(),
                prior_coach_responses=list(self.prior_responses),
            )
            req = ResponseRequest(
                coaching_action=dummy_action,
                topic=topic or DEBATE_TOPICS[0],
                user_position=(position or "For").lower(),
                prior_responses=list(self.prior_responses),
                turn_number=self.turn_count,
            )
            response = loop.run_until_complete(
                self.response_gen.generate_response(req)
            )
            loop.close()
            # Update last_coach_question so it isn't repeated next turn either
            self.last_coach_question = response.text
            return f"[Coach]\n\n{response.text}"
        except Exception as e:
            logger.error(f"Meta question handler failed: {e}")
            return "[Coach]\n\nYou're right, you did address that — let's move on. Can you tell me more about the harm you mentioned?"


# ── Theme: Soft + indigo accent (frozen) ──────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


# ── UI builder ────────────────────────────────────────────────────────────────
def create_interface():
    ui = SpeakFlowUI()

    with gr.Blocks(title="SpeakFlow AI — Debate Coach") as interface:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("# 🎙 SpeakFlow AI — Debate Coach")
        gr.Markdown("Practice structured English debate. Get real-time argument and pronunciation feedback.")

        # ── Row 1: Topic + Position ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=4):
                topic_dropdown = gr.Dropdown(
                    label="Debate Topic",
                    choices=DEBATE_TOPICS,
                    value=DEBATE_TOPICS[0],
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=200):
                position_radio = gr.Radio(
                    label="Your Position",
                    choices=["For", "Against"],
                    value="For",
                    interactive=True,
                )

        # ── Row 2: Input (left) + Output (right) ─────────────────────────────
        with gr.Row():

            # LEFT — input controls
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="✏️ Record Audio",
                    sources=["microphone"],
                    type="filepath",
                )
                typed_input = gr.Textbox(
                    label="Or type transcript here (for demo)",
                    placeholder="e.g. I believe social media is harmful because...",
                    lines=4,
                    interactive=True,
                )
                with gr.Row():
                    analyze_btn = gr.Button("▶ Analyze Turn", variant="primary", scale=3)
                    reset_btn   = gr.Button("↺ New Session",  variant="secondary", scale=1)

            # RIGHT — live results
            with gr.Column(scale=1):
                transcript_out = gr.Textbox(
                    label="📝 Transcript",
                    lines=4,
                    interactive=False,
                )
                improved_out = gr.Textbox(
                    label="💡 How to say it better",
                    lines=4,
                    interactive=False,
                    placeholder="An improved version of your argument will appear here after analysis...",
                )
                coach_out = gr.Textbox(
                    label="🎓 Coach Response",
                    lines=3,
                    interactive=False,
                )
                summary_out = gr.Textbox(
                    label="📌 Language Tips",
                    lines=2,
                    interactive=False,
                    placeholder="Specific language and logic tips will appear here after analysis...",
                )

        # ── Row 3: Status bar ─────────────────────────────────────────────────
        with gr.Row():
            status_out = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=1,
            )

        # ── Row 4: Argument Analysis ──────────────────────────────────────────
        gr.Markdown("### Argument Analysis")
        with gr.Row():
            claim_check    = gr.Checkbox(label="Clarity",   interactive=False)
            reason_check   = gr.Checkbox(label="Reasoning", interactive=False)
            evidence_check = gr.Checkbox(label="Depth",     interactive=False)
            score_out      = gr.Checkbox(label="Fluency",   interactive=False)

        # ── Row 5: Wrap-up options (shown after turn 3) ──────────────────────
        with gr.Group(visible=False) as wrapup_row:
            gr.Markdown("### 🏁 Round Complete — What's next?")
            with gr.Row() as wrapup_btns:
                continue_btn  = gr.Button("▶ Continue this topic", variant="secondary")
                new_topic_btn = gr.Button("🔀 New topic",          variant="secondary")
                stop_btn      = gr.Button("⏹ Stop & get feedback", variant="primary")

        # ── Row 6: Session Summary (shown only after Stop) ───────────────────
        with gr.Group(visible=False) as session_summary_row:
            gr.Markdown("## 📊 Session Summary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎓 Debate Feedback")
                    debate_summary_out = gr.Textbox(
                        label="Overall Argument Feedback",
                        lines=5,
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("### 📌 Language Patterns")
                    lang_summary_out = gr.Textbox(
                        label="Language Summary",
                        lines=5,
                        interactive=False,
                    )

        # ── Row 7: Pronunciation Feedback ─────────────────────────────────────
        gr.Markdown("### Pronunciation Feedback")
        gr.Markdown("_⚠️ Pronunciation analysis is currently in stub mode (MFA not yet integrated). Results will always show no errors until MFA is enabled._")
        pronunciation_out = gr.Textbox(
            label="Mispronounced Words (IPA)",
            lines=2,
            interactive=False,
        )

        # ── Wire up buttons ───────────────────────────────────────────────────
        ALL_OUTPUTS = [
            audio_input,
            transcript_out, coach_out, improved_out, status_out,
            claim_check, reason_check, evidence_check,
            score_out, summary_out, pronunciation_out,
            wrapup_row,
            session_summary_row, debate_summary_out, lang_summary_out,
            analyze_btn,
        ]

        analyze_btn.click(
            fn=ui.analyze_turn,
            inputs=[audio_input, typed_input, topic_dropdown, position_radio],
            outputs=ALL_OUTPUTS,
        )

        reset_btn.click(
            fn=ui.reset_session,
            inputs=[],
            outputs=ALL_OUTPUTS,
        )

        # Wrap-up button handlers
        continue_btn.click(
            fn=ui.continue_session,
            inputs=[],
            outputs=ALL_OUTPUTS,
        )
        new_topic_btn.click(
            fn=ui.reset_session,
            inputs=[],
            outputs=ALL_OUTPUTS,
        )
        stop_btn.click(
            fn=ui.stop_session,
            inputs=[],
            outputs=[wrapup_row, session_summary_row, debate_summary_out, lang_summary_out],
        )

    return interface


if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True, share=False, theme=THEME)