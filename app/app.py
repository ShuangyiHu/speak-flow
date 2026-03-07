import gradio as gr
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['USE_STUB_MFA'] = 'True'

from turn_analyzer import TurnAnalyzer, TurnInput
from coach_policy import CoachPolicyAgent, SessionContext, CoachingStrategy

# Load API key — tries env var first, then .env file at project root
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                ANTHROPIC_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file or run: export ANTHROPIC_API_KEY=sk-...")

# Initialize Whisper (lazy load to avoid slow startup)
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper."""
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return ""

turn_analyzer = TurnAnalyzer(anthropic_api_key=ANTHROPIC_API_KEY)
coach_agent = CoachPolicyAgent(anthropic_api_key=ANTHROPIC_API_KEY)

# In-memory session state
session_state = {
    "session_id": "demo-session",
    "turn_number": 0,
    "coaching_history": [],
    "argument_scores": [],
}

DEBATE_TOPICS = [
    "Social media does more harm than good",
    "Artificial intelligence will eliminate more jobs than it creates",
    "Online education is better than traditional classroom learning",
    "Smartphones should be banned in schools",
    "Remote work is better than working in an office",
]

async def analyze_turn(audio_file, topic, position, transcript_override):
    if not topic:
        return "Please select a debate topic first.", "", False, False, False, 0.0, "", "", ""

    # Determine transcript: manual input takes priority, then Whisper from audio
    transcript = transcript_override.strip() if transcript_override.strip() else ""
    if not transcript:
        if audio_file is not None:
            transcript = transcribe_audio(audio_file)
            if not transcript:
                return "Could not transcribe audio. Try speaking louder or type your argument below.", "", False, False, False, 0.0, "", "", ""
        else:
            return "Please record audio or type a transcript to test.", "", False, False, False, 0.0, "", "", ""

    session_state["turn_number"] += 1

    turn_input = TurnInput(
        transcript=transcript,
        session_id=session_state["session_id"],
        turn_number=session_state["turn_number"],
        topic=topic,
        user_position=position.lower(),
        audio_path=audio_file or "",
        prior_turns=[]
    )

    analysis = await turn_analyzer.analyze(turn_input)

    if analysis is None:
        return "Analysis failed — check your API key and try again.", "", False, False, False, 0.0, "", "", ""

    arg = analysis.argument
    pron = analysis.pronunciation

    session_state["argument_scores"].append(arg.argument_score)

    context = SessionContext(
        session_id=session_state["session_id"],
        topic=topic,
        user_position=position.lower(),
        turn_number=session_state["turn_number"],
        coaching_history=list(session_state["coaching_history"]),
        argument_scores=list(session_state["argument_scores"]),
    )

    action = await coach_agent.decide(analysis, context)
    session_state["coaching_history"].append(action.strategy)

    pron_display = ""
    if pron.mispronounced_words:
        lines = [f"• {e.word}: {e.actual_ipa} → {e.expected_ipa} ({e.severity})" for e in pron.mispronounced_words]
        pron_display = "\n".join(lines)
    else:
        pron_display = "✓ No pronunciation issues detected"

    strategy_label = action.strategy.value.replace("_", " ").title()
    coach_display = f"[{strategy_label}]\n\n{action.response_text}"

    difficulty_map = {1: "📈 Raising difficulty", -1: "📉 Lowering difficulty", 0: ""}
    difficulty_note = difficulty_map.get(action.difficulty_delta, "")

    return (
        transcript,
        coach_display,
        arg.has_claim,
        arg.has_reasoning,
        arg.has_evidence,
        round(arg.argument_score, 2),
        arg.summary,
        pron_display,
        difficulty_note
    )

def reset_session():
    session_state["turn_number"] = 0
    session_state["coaching_history"] = []
    session_state["argument_scores"] = []
    return [None, "", "", False, False, False, 0.0, "", "", ""]

def run_analyze(audio_file, topic, position, transcript_override):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, analyze_turn(audio_file, topic, position, transcript_override))
                return future.result()
        else:
            return loop.run_until_complete(analyze_turn(audio_file, topic, position, transcript_override))
    except RuntimeError:
        return asyncio.run(analyze_turn(audio_file, topic, position, transcript_override))

with gr.Blocks(title="SpeakFlow AI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ SpeakFlow AI — Debate Coach")
    gr.Markdown("Practice structured English debate. Get real-time argument and pronunciation feedback.")

    with gr.Row():
        with gr.Column(scale=2):
            topic_dropdown = gr.Dropdown(
                choices=DEBATE_TOPICS,
                label="Debate Topic",
                value=DEBATE_TOPICS[0]
            )
        with gr.Column(scale=1):
            position_radio = gr.Radio(
                choices=["For", "Against"],
                label="Your Position",
                value="For"
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Audio"
            )
            transcript_input = gr.Textbox(
                label="Or type transcript here (for demo)",
                placeholder="e.g. I believe social media is harmful because...",
                lines=2
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze Turn", variant="primary", scale=3)
                reset_btn = gr.Button("Reset Session", scale=1)

        with gr.Column():
            transcript_out = gr.Textbox(label="Transcript", lines=3, interactive=False)
            coach_out = gr.Textbox(label="🤖 Coach Response", lines=4, interactive=False)
            difficulty_out = gr.Textbox(label="Difficulty Adjustment", interactive=False)

    gr.Markdown("### Argument Analysis")
    with gr.Row():
        claim_check = gr.Checkbox(label="Has Claim", interactive=False)
        reason_check = gr.Checkbox(label="Has Reasoning", interactive=False)
        evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)

    with gr.Row():
        score_out = gr.Number(label="Argument Score (0–1)", interactive=False)
        summary_out = gr.Textbox(label="Summary", interactive=False, scale=3)

    gr.Markdown("### Pronunciation Feedback")
    pronunciation_out = gr.Textbox(label="Mispronounced Words (IPA)", lines=3, interactive=False)

    analyze_btn.click(
        fn=run_analyze,
        inputs=[audio_input, topic_dropdown, position_radio, transcript_input],
        outputs=[
            transcript_out, coach_out,
            claim_check, reason_check, evidence_check,
            score_out, summary_out,
            pronunciation_out, difficulty_out
        ]
    )

    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[
            audio_input, transcript_input, transcript_out,
            claim_check, reason_check, evidence_check,
            score_out, summary_out, pronunciation_out, difficulty_out
        ]
    )

if __name__ == "__main__":
    app.launch()