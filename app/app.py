"""
app.py — SpeakFlow AI Gradio frontend
======================================
⚠️  UI DESIGN IS FROZEN — DO NOT MODIFY LAYOUT WITHOUT EXPLICIT APPROVAL ⚠️

Latency optimizations applied (v3):
  OPT-1: RAG HyDE Claude call removed — replaced with template string (~2s saved)
  OPT-2: CoachPolicyAgent + RAGRetriever parallelized via asyncio.gather (~2.5s saved)
  OPT-3: ImprovedVersion + LanguageTips parallelized via asyncio.gather (~2.5s saved)
  Expected total saving: ~7s per turn vs serial baseline (~17s → ~10s)

Timing instrumentation:
  Each pipeline step logs: [TIMING] step=<n> duration=<Xs> cumulative=<Xs>
  Final log line: [TIMING SUMMARY] total=<Xs> breakdown=<dict>
  Compare baseline vs optimized by grepping: grep "TIMING" app.log
"""

import asyncio
import os
import time
import logging

import gradio as gr
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from turn_analyzer import TurnAnalyzer
from coach_policy import CoachPolicyAgent
from response_generator import ResponseGenerator
from rag_retriever import RAGRetriever
from shared_types import (
    TurnInput,
    CoachingStrategy,
    CoachingAction,
    SessionContext,
    TurnIntent,
    ResponseRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("USE_STUB_MFA", "True")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Whisper lazy-load ─────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            model_name = os.getenv("WHISPER_MODEL", "base")
            logger.info(f"Loading Whisper model: {model_name}")
            _whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.error("openai-whisper not installed.")
            _whisper_model = None
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""


DEBATE_TOPICS = [
    "Social media does more harm than good",
    "AI will take more jobs than it creates",
    "Climate change policy should prioritise economic growth",
    "Standardised testing is an effective measure of student ability",
    "Universal basic income should be implemented globally",
    "Space exploration funding should be redirected to solving poverty",
    "Renewable energy can fully replace fossil fuels by 2050",
]


# ── Timing helper ─────────────────────────────────────────────────────────────
class StepTimer:
    """Accumulates per-step durations and emits structured TIMING log lines.

    Usage:
        timer = StepTimer(turn_number)
        ...do work...
        timer.mark("step_name")   # logs duration + cumulative
        timer.summary()           # logs full breakdown
    """

    def __init__(self, turn: int):
        self.turn = turn
        self.t0   = time.time()
        self.prev = self.t0
        self.steps: dict[str, float] = {}

    def mark(self, name: str) -> float:
        now       = time.time()
        duration  = now - self.prev
        cumul     = now - self.t0
        self.steps[name] = round(duration, 3)
        self.prev = now
        logger.info(
            f"[TIMING] turn={self.turn} step={name} "
            f"duration={duration:.3f}s cumulative={cumul:.3f}s"
        )
        return duration

    def summary(self) -> float:
        total = time.time() - self.t0
        logger.info(
            f"[TIMING SUMMARY] turn={self.turn} total={total:.3f}s "
            f"breakdown={self.steps}"
        )
        return total


# ── State class ───────────────────────────────────────────────────────────────
class SpeakFlowUI:
    def __init__(self):
        self.turn_analyzer  = TurnAnalyzer(anthropic_api_key=ANTHROPIC_API_KEY)
        self.coach          = CoachPolicyAgent(mfa_enabled=False)
        self.response_gen   = ResponseGenerator()
        self.rag_retriever  = RAGRetriever()

        self.turn_count: int              = 0
        self.prior_turns: list[str]       = []
        self.prior_responses: list[str]   = []
        self.argument_scores: list[float] = []
        self.coaching_history: list[CoachingStrategy] = []
        self.last_coach_question: str     = ""
        self.last_turn_intent: str        = ""
        self.session_transcripts: list[str]   = []
        self.session_language_tips: list[str] = []
        self.session_topic: str    = ""
        self.session_position: str = ""
        self.in_wrapup: bool       = False

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
            None, "", "", "",
            "🔄 Session reset. Record your first argument.",
            False, False, False, False, "", "",
            gr.update(visible=False),
            gr.update(visible=False), "", "",
            gr.update(interactive=True),
        )

    def continue_session(self):
        self.turn_count = 2
        self.in_wrapup = False
        return (
            None, "", "", "",
            "▶ Continuing — record your next argument above.",
            False, False, False, False, "", "",
            gr.update(visible=False),
            gr.update(visible=False), "", "",
            gr.update(interactive=True),
        )

    async def stop_session(self):
        self.in_wrapup = True
        debate_summary = "Great session! You built up your argument well across the turns."
        lang_summary   = "Keep practising connectors like 'therefore' and 'however'."

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

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            debate_summary,
            lang_summary,
        )

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze_turn(self, audio_file, typed_transcript, topic, position):
        """
        Optimized pipeline (v3):
          1.  Whisper / typed                             [local]
          2.  TurnAnalyzer                                [Claude API]
          3.  CoachPolicyAgent + RAGRetriever             [PARALLEL — OPT-2]
          4.  ResponseGenerator                           [Claude API]
          5.  ImprovedVersion + LanguageTips              [PARALLEL — OPT-3]
        RAG HyDE no longer calls Claude (template string) [OPT-1]

        Every step is timed and logged as [TIMING] for before/after comparison.
        """
        timer = StepTimer(self.turn_count + 1)

        if self.in_wrapup:
            return (
                None, "", "", "",
                "⏸ Round complete — please choose Continue, New Topic, or Stop below.",
                False, False, False, False, "", "",
                gr.update(visible=True),
                gr.update(visible=False), "", "",
                gr.update(interactive=False),
            )

        self.session_topic    = topic or DEBATE_TOPICS[0]
        self.session_position = (position or "For").lower()

        # ── Step 1: Transcript ────────────────────────────────────────────────
        if typed_transcript and typed_transcript.strip():
            transcript = typed_transcript.strip()
            transcript_source = "typed"
            timer.mark("1_transcript_typed")
        elif audio_file is not None:
            transcript = transcribe_audio(audio_file)
            timer.mark("1_transcript_whisper")
            if not transcript:
                return (
                    None, "", "⚠️ Transcription failed.", "",
                    "❌ Transcription error.", False, False, False, False, "", "",
                    gr.update(visible=False),
                    gr.update(visible=False), "", "", gr.update(interactive=True),
                )
            transcript_source = "whisper"
        else:
            return (
                None, "", "Please record audio or type a transcript first.", "",
                "⚠️ No input provided.", False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        self.turn_count += 1

        # ── Step 2: Intent detection (local, fast) ────────────────────────────
        intent = self.turn_analyzer._detect_intent(transcript)
        self.last_turn_intent = intent.value

        if intent == TurnIntent.META_QUESTION:
            meta_response = self._handle_meta_question(transcript, topic, position)
            timer.mark("2_meta_question")
            timer.summary()
            status = f"💬 Turn {self.turn_count} — Coach answered your question."
            return (
                None, transcript, meta_response, "", status,
                False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        # ── Step 3: TurnAnalyzer (Claude API) ────────────────────────────────
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
            analysis = loop.run_until_complete(self.turn_analyzer.analyze(turn_input))
            timer.mark("3_turn_analyzer")
        except Exception as e:
            logger.error(f"TurnAnalyzer failed: {e}")
            timer.mark("3_turn_analyzer_FAILED")
            return (
                None, transcript, f"⚠️ Analysis failed: {e}", "",
                f"❌ Turn {self.turn_count} — analysis error.",
                False, False, False, False, "", "",
                gr.update(visible=False),
                gr.update(visible=False), "", "", gr.update(interactive=True),
            )

        self.prior_turns.append(analysis.argument.summary or transcript)
        self.argument_scores.append(analysis.argument.argument_score)
        self.session_transcripts.append(transcript)

        # ── Step 4: SessionContext (local, instant) ───────────────────────────
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

        # ── Step 5: CoachPolicyAgent + RAG PARALLEL  [OPT-2] ─────────────────
        # RAG runs with PROBE pre-action so it doesn't wait for CoachPolicy.
        # Strategy-based type filtering is a nice-to-have; correctness is fine
        # without it since re-ranking still selects the best chunks.
        pre_rag_action = CoachingAction(
            strategy=CoachingStrategy.PROBE,
            topic=topic or DEBATE_TOPICS[0],
            user_position=(position or "For").lower(),
            intent="",
            target_claim=None,
            target_word=None,
            target_phoneme=None,
            argument_score=analysis.argument.argument_score,
            pronunciation_score=1.0,
            difficulty_delta=0,
            turn_number=self.turn_count,
            prior_coach_responses=list(self.prior_responses),
        )

        async def _parallel_coach_rag():
            return await asyncio.gather(
                self.coach.decide(analysis, session_ctx),
                self.rag_retriever.retrieve(pre_rag_action, analysis),
                return_exceptions=True,
            )

        coach_result, rag_result = loop.run_until_complete(_parallel_coach_rag())
        timer.mark("5_coach+rag_parallel")

        coaching_action = None if isinstance(coach_result, Exception) else coach_result
        retrieval_ctx   = None if isinstance(rag_result,   Exception) else rag_result

        if isinstance(coach_result, Exception):
            logger.error(f"CoachPolicyAgent failed: {coach_result}")
        if isinstance(rag_result, Exception):
            logger.error(f"RAGRetriever failed: {rag_result}")

        if retrieval_ctx:
            logger.info(
                f"[RAG] turn={self.turn_count} "
                f"fallback={retrieval_ctx.fallback_used} "
                f"chunks={len(retrieval_ctx.chunks)} "
                f"rag_latency={retrieval_ctx.retrieval_latency_ms}ms "
                f"hyde='{retrieval_ctx.hypothetical_query[:80]}'"
            )

        # Enrich coaching_action.intent for ResponseGenerator
        if coaching_action:
            self.coaching_history.append(coaching_action.strategy)
            arg = analysis.argument
            dim_lines = [
                f"  {label}: {val:.1f} {'✓' if val >= thr else '✗'}"
                for label, val, thr in [
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
            coaching_action.intent = "\n".join([
                f"Topic: {topic or DEBATE_TOPICS[0]}",
                f"Student position: {(position or 'For').lower()}",
                f"Turn {self.turn_count}. Student just said:",
                f'  "{transcript[:300]}"',
                "Scores:", *dim_lines,
                f"Coach summary: {arg.summary}",
                f"Weakest areas: {', '.join(weakest) if weakest else 'none — strong'}",
                f"Coach last asked: \"{self.last_coach_question}\"",
                "Do NOT repeat or rephrase the coach's last question.",
            ])

        # ── Step 6: ResponseGenerator (Claude API) ────────────────────────────
        coach_text = ""
        if coaching_action:
            request = ResponseRequest(
                coaching_action=coaching_action,
                topic=topic or DEBATE_TOPICS[0],
                user_position=(position or "For").lower(),
                prior_responses=list(self.prior_responses),
                turn_number=self.turn_count,
                retrieval_context=retrieval_ctx,
            )
            try:
                response = loop.run_until_complete(
                    self.response_gen.generate_response(request)
                )
                strategy_label = coaching_action.strategy.value.title()
                coach_text = f"[{strategy_label}]\n\n{response.text}"
                self.last_coach_question = response.text
                self.prior_responses.append(response.text)
                if len(self.prior_responses) > 5:
                    self.prior_responses.pop(0)
            except Exception as e:
                logger.error(f"ResponseGenerator failed: {e}")
                coach_text = f"[{coaching_action.strategy.value.title()}]\n\n(unavailable: {e})"
            timer.mark("6_response_generator")

        # ── Step 7: ImprovedVersion + LanguageTips PARALLEL  [OPT-3] ──────────
        improved_text = ""
        language_tips = ""
        if coaching_action:
            async def _parallel_secondary():
                return await asyncio.gather(
                    self.response_gen.generate_improved_version(
                        original_transcript=transcript,
                        topic=topic or DEBATE_TOPICS[0],
                        user_position=(position or "For").lower(),
                        vocabulary_flags=analysis.argument.vocabulary_flags,
                    ),
                    self.response_gen.generate_language_tips(
                        original_transcript=transcript,
                        improved_transcript="",
                        vocabulary_flags=analysis.argument.vocabulary_flags,
                        clarity_feedback=analysis.argument.clarity_feedback or "",
                        fluency_feedback=analysis.argument.fluency_feedback or "",
                    ),
                    return_exceptions=True,
                )

            sec = loop.run_until_complete(_parallel_secondary())
            timer.mark("7_improved+tips_parallel")

            improved_text = sec[0] if not isinstance(sec[0], Exception) else ""
            language_tips = sec[1] if not isinstance(sec[1], Exception) else ""
            if isinstance(sec[0], Exception):
                logger.error(f"ImprovedVersion failed: {sec[0]}")
            if isinstance(sec[1], Exception):
                logger.error(f"LanguageTips failed: {sec[1]}")

        loop.close()
        total = timer.summary()

        self.session_language_tips.append(language_tips)
        show_wrapup = self.turn_count >= 3
        if show_wrapup:
            self.in_wrapup = True

        errors = analysis.pronunciation.mispronounced_words
        pronunciation_text = (
            "\n".join(
                f"• {w.word}  expected: {w.expected_ipa}  actual: {w.actual_ipa}  [{w.severity.value}]"
                for w in errors
            ) if errors else "✓ No pronunciation issues detected"
        )

        arg   = analysis.argument
        score = arg.argument_score
        emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        status = (
            f"✅ Turn {self.turn_count} complete  "
            f"{emoji} Score: {score:.2f}  "
            f"⏱ {total:.1f}s ({transcript_source})"
            + (" — Choose an option below" if show_wrapup
               else " — Record your next argument above")
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
            gr.update(visible=False), "", "",
            gr.update(interactive=not show_wrapup),
        )

    # ── Meta-question handler ─────────────────────────────────────────────────
    def _handle_meta_question(self, transcript: str, topic: str, position: str) -> str:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dummy_action = CoachingAction(
                strategy=CoachingStrategy.PROBE,
                intent=(
                    f"Coach previously asked: \"{self.last_coach_question}\"\n"
                    f"Student pushed back: \"{transcript}\"\n"
                    f"Student is RIGHT. Acknowledge it, do NOT repeat same question, "
                    f"ask ONE new forward-moving question."
                ),
                target_claim=None, target_word=None, target_phoneme=None,
                argument_score=0.0, pronunciation_score=1.0, difficulty_delta=0,
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
            response = loop.run_until_complete(self.response_gen.generate_response(req))
            loop.close()
            self.last_coach_question = response.text
            return f"[Coach]\n\n{response.text}"
        except Exception as e:
            logger.error(f"Meta question handler failed: {e}")
            return "[Coach]\n\nYou're right — let's move on. What specific harm can you elaborate on?"


# ── Theme ─────────────────────────────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


# ── UI ────────────────────────────────────────────────────────────────────────
def create_interface():
    ui = SpeakFlowUI()

    with gr.Blocks(title="SpeakFlow AI — Debate Coach", theme=THEME) as interface:

        gr.Markdown("# 🎙 SpeakFlow AI — Debate Coach")
        gr.Markdown("Practice structured English debate. Get real-time argument and pronunciation feedback.")

        with gr.Row():
            with gr.Column(scale=4):
                topic_dropdown = gr.Dropdown(
                    label="Debate Topic", choices=DEBATE_TOPICS,
                    value=DEBATE_TOPICS[0], interactive=True,
                )
            with gr.Column(scale=1, min_width=200):
                position_radio = gr.Radio(
                    label="Your Position", choices=["For", "Against"],
                    value="For", interactive=True,
                )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="✏️ Record Audio", sources=["microphone"], type="filepath",
                )
                typed_input = gr.Textbox(
                    label="Or type transcript here (for demo)",
                    placeholder="e.g. I believe social media is harmful because...",
                    lines=4, interactive=True,
                )
                with gr.Row():
                    analyze_btn = gr.Button("▶ Analyze Turn", variant="primary", scale=3)
                    reset_btn   = gr.Button("↺ New Session",  variant="secondary", scale=1)

            with gr.Column(scale=1):
                transcript_out = gr.Textbox(label="📝 Transcript",        lines=4, interactive=False)
                improved_out   = gr.Textbox(
                    label="💡 How to say it better", lines=4, interactive=False,
                    placeholder="An improved version of your argument will appear here...",
                )
                coach_out    = gr.Textbox(label="🎓 Coach Response",       lines=3, interactive=False)
                summary_out  = gr.Textbox(
                    label="📌 Language Tips", lines=2, interactive=False,
                    placeholder="Language tips will appear here after analysis...",
                )

        with gr.Row():
            status_out = gr.Textbox(label="Status", interactive=False, max_lines=1)

        gr.Markdown("### Argument Analysis")
        with gr.Row():
            claim_check    = gr.Checkbox(label="Clarity",   interactive=False)
            reason_check   = gr.Checkbox(label="Reasoning", interactive=False)
            evidence_check = gr.Checkbox(label="Depth",     interactive=False)
            score_out      = gr.Checkbox(label="Fluency",   interactive=False)

        with gr.Group(visible=False) as wrapup_row:
            gr.Markdown("### 🏁 Round Complete — What's next?")
            with gr.Row():
                continue_btn  = gr.Button("▶ Continue this topic", variant="secondary")
                new_topic_btn = gr.Button("🔀 New topic",          variant="secondary")
                stop_btn      = gr.Button("⏹ Stop & get feedback", variant="primary")

        with gr.Group(visible=False) as session_summary_row:
            gr.Markdown("## 📊 Session Summary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎓 Debate Feedback")
                    debate_summary_out = gr.Textbox(
                        label="Overall Argument Feedback", lines=5, interactive=False)
                with gr.Column():
                    gr.Markdown("### 📌 Language Patterns")
                    lang_summary_out = gr.Textbox(
                        label="Language Summary", lines=5, interactive=False)

        gr.Markdown("### Pronunciation Feedback")
        gr.Markdown("_⚠️ Pronunciation analysis is currently in stub mode (MFA not yet integrated)._")
        pronunciation_out = gr.Textbox(
            label="Mispronounced Words (IPA)", lines=2, interactive=False)

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
        reset_btn.click(fn=ui.reset_session,      inputs=[], outputs=ALL_OUTPUTS)
        continue_btn.click(fn=ui.continue_session, inputs=[], outputs=ALL_OUTPUTS)
        new_topic_btn.click(fn=ui.reset_session,   inputs=[], outputs=ALL_OUTPUTS)
        stop_btn.click(
            fn=ui.stop_session, inputs=[],
            outputs=[wrapup_row, session_summary_row, debate_summary_out, lang_summary_out],
        )

    return interface


if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True, share=False, theme=THEME)