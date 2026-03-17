"""
app.py — SpeakFlow AI Gradio frontend  (v10)
============================================
⚠️  UI DESIGN IS FROZEN — DO NOT MODIFY LAYOUT WITHOUT EXPLICIT APPROVAL ⚠️

Architecture:
  All per-turn orchestration (Steps 3-7) is now delegated to pipeline.py
  (LangGraph StateGraph). app.py is responsible only for UI and session
  lifecycle — not for wiring modules together.

  pipeline.ainvoke() runs the full graph per turn:
    intent_node → fan-out [score_node ‖ summary_node ‖ pronunciation_node]
    → merge_analysis_node → [coach_policy_node ‖ rag_node] → response_node
    → update_session_node

  Session state (prior_turns, coaching_history, argument_scores, etc.) is
  persisted by LangGraph MemorySaver, keyed by thread_id = session_id.
  reset_session() uses a new thread_id to start a clean slate.

Key UI decisions:
  - Wrapup buttons always rendered, start as interactive=False (Gradio 6.x
    visible= updates on hidden components are unreliable).
  - debate_summary_out / lang_summary_out excluded from ALL_OUTPUTS to
    prevent loading spinners during per-turn analysis.
  - stop_session generates summaries via direct AsyncAnthropic calls (45s).
  - PronunciationCoach runs AFTER pipeline.ainvoke(), using turn_analysis
    from pipeline result. In stub MFA mode, mispronounced_words is always
    empty so PronunciationCoach returns instantly with no LLM call.
"""

import asyncio
import os
import time
import logging

import gradio as gr
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from pipeline import SpeakFlowPipeline
from pronunciation_coach import PronunciationCoach

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("USE_STUB_MFA", "True")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SESSION_SUMMARY_TIMEOUT = 45.0

DEBATE_TOPICS = [
    "Social media does more harm than good",
    "AI will take more jobs than it creates",
    "Climate change policy should prioritise economic growth",
    "Standardised testing is an effective measure of student ability",
    "Universal basic income should be implemented globally",
    "Space exploration funding should be redirected to solving poverty",
    "Renewable energy can fully replace fossil fuels by 2050",
]

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


# ── Timing helper ─────────────────────────────────────────────────────────────
class StepTimer:
    def __init__(self, turn: int):
        self.turn = turn
        self.t0   = time.time()
        self.prev = self.t0
        self.steps: dict[str, float] = {}

    def mark(self, name: str) -> float:
        now      = time.time()
        duration = now - self.prev
        cumul    = now - self.t0
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


# ── Pronunciation text formatter ──────────────────────────────────────────────
def _format_pronunciation_feedback(feedback) -> str:
    """
    Convert PronunciationFeedback → display string for pronunciation_out textbox.
    Called only when PronunciationCoach returns a real result.
    """
    lines = [feedback.overall_message]

    if feedback.has_errors:
        for c in feedback.corrections:
            lines.append(f"\n• {c.word}  [{c.severity.value}]")
            lines.append(f"  ↳ {c.correction_tip}")
            lines.append(f"  e.g. \"{c.model_sentence}\"")
        if feedback.drill_sentence:
            lines.append(f"\n🗣 Drill: {feedback.drill_sentence}")

    lines.append(f"\n{feedback.fluency_comment}")
    return "\n".join(lines)


# ── State class ───────────────────────────────────────────────────────────────
class SpeakFlowUI:
    def __init__(self):
        # Pipeline owns all module instances internally (TurnAnalyzer,
        # CoachPolicyAgent, ResponseGenerator, RAGRetriever).
        # ResponseGenerator is still exposed for stop_session summaries.
        self.pipeline           = SpeakFlowPipeline()
        self.response_gen       = self.pipeline._response_gen   # reuse for stop_session
        self.pronunciation_coach = PronunciationCoach(
            anthropic_api_key=ANTHROPIC_API_KEY
        )
        self._session_id  = "gradio-session-0"

        # Local state: only what pipeline.py does NOT persist
        self.session_topic: str    = ""
        self.session_position: str = ""
        self.in_wrapup: bool       = False

    @staticmethod
    def _run(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ── Wrapup button state helpers ───────────────────────────────────────────
    @staticmethod
    def _wrapup_active(active: bool):
        return (
            gr.update(interactive=active),
            gr.update(interactive=active),
            gr.update(interactive=active),
        )

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset_session(self):
        import uuid
        self._session_id      = f"gradio-session-{uuid.uuid4().hex[:8]}"
        self.session_topic    = ""
        self.session_position = ""
        self.in_wrapup        = False
        c, n, s = self._wrapup_active(False)
        return (
            None, "", "", "",
            "🔄 Session reset. Record your first argument.",
            False, False, False, False, "", "",
            c, n, s,
            gr.update(interactive=True),
            "", "",
        )

    def continue_session(self):
        self.in_wrapup = False
        c, n, s = self._wrapup_active(False)
        return (
            None, "", "", "",
            "▶ Continuing — record your next argument above.",
            False, False, False, False, "", "",
            c, n, s,
            gr.update(interactive=True),
            "", "",
        )

    # ── stop_session ──────────────────────────────────────────────────────────
    def stop_session(self):
        """Generate summaries via direct Claude API calls (45s budget).
        Session data (transcripts, scores, tips) is read from pipeline MemorySaver.
        Returns updates for: continue_btn, new_topic_btn, stop_btn,
                              debate_summary_out, lang_summary_out
        """
        self.in_wrapup = True
        t_start = time.time()

        debate_summary = "Great session! You built up your argument well across the turns."
        lang_summary   = "Keep practising connectors like 'therefore' and 'however'."

        config = {"configurable": {"thread_id": self._session_id}}
        session_state = self.pipeline.get_session_state(config)
        session_transcripts   = session_state.get("session_transcripts")   or []
        argument_scores       = session_state.get("argument_scores")       or []
        session_language_tips = session_state.get("session_language_tips") or []

        if session_transcripts:
            async def _summaries_direct():
                client = self.response_gen._client
                model  = self.response_gen._model

                turns_text = "\n".join(
                    f"Turn {i+1} (score {argument_scores[i]:.2f}): {t}"
                    for i, t in enumerate(session_transcripts)
                    if i < len(argument_scores)
                )
                avg = (sum(argument_scores) / len(argument_scores)
                       if argument_scores else 0)

                debate_prompt = (
                    f'You are a debate coach giving end-of-session feedback.\n\n'
                    f'Topic: "{self.session_topic}" | Position: {self.session_position}\n\n'
                    f'Turns:\n{turns_text}\n\n'
                    f'Avg score: {avg:.2f}/1.0\n\n'
                    'Write 3-4 sentences: one specific strength (name a moment), '
                    'one improvement area, one encouraging close. Plain text only.'
                )
                tips_text = "\n".join(
                    f"Turn {i+1}: {t}"
                    for i, t in enumerate(session_language_tips) if t
                )
                lang_prompt = (
                    'You are an English teacher summarising language patterns.\n\n'
                    f'Transcripts:\n{turns_text}\n\n'
                    f'Per-turn tips:\n{tips_text or "none"}\n\n'
                    'Write 2 sentences: one recurring language pattern, '
                    'one concrete practice suggestion. Plain text only.'
                )

                d_msg, l_msg = await asyncio.gather(
                    client.messages.create(
                        model=model, max_tokens=300,
                        messages=[{"role": "user", "content": debate_prompt}]
                    ),
                    client.messages.create(
                        model=model, max_tokens=200,
                        messages=[{"role": "user", "content": lang_prompt}]
                    ),
                    return_exceptions=True,
                )
                return d_msg, l_msg

            try:
                d_msg, l_msg = self._run(
                    asyncio.wait_for(_summaries_direct(), timeout=SESSION_SUMMARY_TIMEOUT)
                )
                if not isinstance(d_msg, Exception):
                    debate_summary = d_msg.content[0].text.strip()
                else:
                    logger.error(f"Debate summary failed: {d_msg}")
                if not isinstance(l_msg, Exception):
                    lang_summary = l_msg.content[0].text.strip()
                else:
                    logger.error(f"Lang summary failed: {l_msg}")
            except asyncio.TimeoutError:
                logger.error(f"stop_session timed out after {SESSION_SUMMARY_TIMEOUT}s")
            except Exception as e:
                logger.error(f"stop_session error: {e}")

        elapsed = time.time() - t_start
        logger.info(f"[STOP] done in {elapsed:.1f}s | debate: {debate_summary[:60]}")

        c, n, s = self._wrapup_active(False)
        return c, n, s, debate_summary, lang_summary

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze_turn(self, audio_file, typed_transcript, topic, position):
        timer = StepTimer(1)

        if self.in_wrapup:
            c, n, s = self._wrapup_active(True)
            return (
                None, "", "", "",
                "⏸ Round complete — choose Continue, New Topic, or Stop below.",
                False, False, False, False, "", "",
                c, n, s,
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
                c, n, s = self._wrapup_active(False)
                return (
                    None, "", "⚠️ Transcription failed.", "",
                    "❌ Transcription error.",
                    False, False, False, False, "", "",
                    c, n, s, gr.update(interactive=True),
                )
            transcript_source = "whisper"
        else:
            c, n, s = self._wrapup_active(False)
            return (
                None, "", "Please record audio or type a transcript first.", "",
                "⚠️ No input.",
                False, False, False, False, "", "",
                c, n, s, gr.update(interactive=True),
            )

        # ── Steps 2-7: delegated to LangGraph pipeline ───────────────────────
        config = {"configurable": {"thread_id": self._session_id}}
        try:
            result = self._run(self.pipeline.ainvoke({
                "transcript":    transcript,
                "topic":         topic or DEBATE_TOPICS[0],
                "user_position": (position or "For").lower(),
                "session_id":    self._session_id,
                "audio_path":    audio_file or "",
            }, config=config))
            timer.mark("pipeline_ainvoke")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            c, n, s = self._wrapup_active(False)
            return (
                None, transcript, f"⚠️ Pipeline error: {e}", "",
                f"❌ Pipeline error.",
                False, False, False, False, "", "",
                c, n, s, gr.update(interactive=True),
            )

        # ── Extract outputs from pipeline state ───────────────────────────────
        coach_text    = result.get("coach_text", "")
        improved_text = result.get("improved_text", "")
        language_tips = result.get("language_tips", "")
        turn_number   = result.get("turn_number", 1)
        analysis      = result.get("turn_analysis")

        # ── Step 8: Pronunciation coaching (post-pipeline) ────────────────────
        # In stub MFA mode: analysis.pronunciation.mispronounced_words == []
        # → PronunciationCoach returns instantly with no LLM call.
        # When real MFA is integrated: pipeline will populate mispronounced_words
        # and this step will call Claude for correction tips.
        pronunciation_text = result.get(
            "pronunciation_text", "✓ No pronunciation issues detected"
        )
        if analysis is not None:
            try:
                pron_feedback = self._run(
                    self.pronunciation_coach.generate_feedback(
                        pronunciation_result=analysis.pronunciation,
                        transcript=transcript,
                        topic=topic or DEBATE_TOPICS[0],
                    )
                )
                pronunciation_text = _format_pronunciation_feedback(pron_feedback)
                timer.mark("8_pronunciation_coach")
                logger.info(
                    f"[PRONUNCIATION] has_errors={pron_feedback.has_errors} "
                    f"latency={pron_feedback.latency_ms}ms"
                )
            except Exception as e:
                logger.error(f"PronunciationCoach failed: {e}")
                # Keep the pipeline fallback text — do not crash the turn

        self.session_topic    = topic or DEBATE_TOPICS[0]
        self.session_position = (position or "For").lower()

        total = timer.summary()

        base_status = result.get("status_message", f"✅ Turn {turn_number} complete")
        status = base_status.split(" — ")[0] + f"  ⏱ {total:.1f}s ({transcript_source})"
        if "Choose an option" in base_status:
            status += " — Choose an option below"
        elif "next argument" in base_status:
            status += " — Record your next argument above"

        show_wrapup = turn_number >= 3
        if show_wrapup:
            self.in_wrapup = True

        c, n, s = self._wrapup_active(show_wrapup)

        if analysis:
            arg = analysis.argument
            return (
                None,
                transcript, coach_text, improved_text, status,
                gr.update(value=arg.clarity_score >= 0.5,     label=f"Clarity  {arg.clarity_score:.1f}"),
                gr.update(value=arg.reasoning_score >= 0.5,   label=f"Reasoning  {arg.reasoning_score:.1f}"),
                gr.update(value=arg.depth_score >= 0.4,       label=f"Depth  {arg.depth_score:.1f}"),
                gr.update(value=arg.fluency_score_arg >= 0.5, label=f"Fluency  {arg.fluency_score_arg:.1f}"),
                language_tips, pronunciation_text,
                c, n, s,
                gr.update(interactive=not show_wrapup),
            )
        else:
            return (
                None,
                transcript, coach_text, improved_text, status,
                False, False, False, False,
                language_tips, pronunciation_text,
                c, n, s,
                gr.update(interactive=True),
            )


# ── UI ────────────────────────────────────────────────────────────────────────
def create_interface():
    ui = SpeakFlowUI()

    with gr.Blocks(title="SpeakFlow AI — Debate Coach") as interface:

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

        gr.Markdown("### 🏁 Round Complete — What's next?")
        with gr.Row():
            continue_btn  = gr.Button("▶ Continue this topic", variant="secondary", interactive=False)
            new_topic_btn = gr.Button("🔀 New topic",           variant="secondary", interactive=False)
            stop_btn      = gr.Button("⏹ Stop & get feedback",  variant="primary",   interactive=False)

        gr.Markdown("### 📊 Session Summary")
        with gr.Row():
            debate_summary_out = gr.Textbox(
                label="🎓 Overall Argument Feedback",
                lines=5, interactive=False, value="",
                placeholder="Session summary will appear here after you stop the session.",
            )
            lang_summary_out = gr.Textbox(
                label="📌 Language Patterns",
                lines=5, interactive=False, value="",
                placeholder="Language patterns will appear here after you stop the session.",
            )

        gr.Markdown("### 🗣 Pronunciation Feedback")
        pronunciation_out = gr.Textbox(
            label="Pronunciation Coaching",
            lines=4, interactive=False,
            placeholder="Pronunciation feedback will appear here after each turn.\n"
                        "(Currently in stub mode — feedback activates when MFA is integrated.)",
        )

        ALL_OUTPUTS = [
            audio_input,          # 0
            transcript_out,       # 1
            coach_out,            # 2
            improved_out,         # 3
            status_out,           # 4
            claim_check,          # 5
            reason_check,         # 6
            evidence_check,       # 7
            score_out,            # 8
            summary_out,          # 9
            pronunciation_out,    # 10
            continue_btn,         # 11
            new_topic_btn,        # 12
            stop_btn,             # 13
            analyze_btn,          # 14
        ]

        STOP_OUTPUTS = [
            continue_btn,         # 0
            new_topic_btn,        # 1
            stop_btn,             # 2
            debate_summary_out,   # 3
            lang_summary_out,     # 4
        ]

        CLEAR_OUTPUTS = ALL_OUTPUTS + [debate_summary_out, lang_summary_out]

        analyze_btn.click(
            fn=ui.analyze_turn,
            inputs=[audio_input, typed_input, topic_dropdown, position_radio],
            outputs=ALL_OUTPUTS,
        )
        reset_btn.click(fn=ui.reset_session,       inputs=[], outputs=CLEAR_OUTPUTS)
        continue_btn.click(fn=ui.continue_session, inputs=[], outputs=CLEAR_OUTPUTS)
        new_topic_btn.click(fn=ui.reset_session,   inputs=[], outputs=CLEAR_OUTPUTS)
        stop_btn.click(
            fn=ui.stop_session, inputs=[],
            outputs=STOP_OUTPUTS,
        )

    return interface


if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True, share=False)