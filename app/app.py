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
import gradio as gr
import asyncio
from pronunciation_coach import PronunciationCoach
from shared_types import WordError, PronunciationResult, ErrorSeverity
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Set stub mode for demo
os.environ["USE_STUB_MFA"] = "true"

class TurnAnalyzer:
    def __init__(self):
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.coach = PronunciationCoach(anthropic_api_key)
    
    def analyze(self, audio_file, topic="general discussion"):
        # Stub transcript for demo
        transcript = "This is a sample transcript from the audio recording."
        
        # Stub pronunciation analysis with sample errors
        sample_errors = [
            WordError(
                word="this",
                expected_ipa="/θɪs/",
                actual_ipa="/dɪs/",
                severity=ErrorSeverity.MEDIUM
            ),
            WordError(
                word="sample",
                expected_ipa="/sæmpl/",
                actual_ipa="/sempl/",
                severity=ErrorSeverity.LOW
            )
        ]
        
        pronunciation_result = PronunciationResult(
            mispronounced_words=sample_errors,
            fluency_score=0.75
        )
        
        # Generate feedback using the coach
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            feedback = loop.run_until_complete(
                self.coach.generate_feedback(pronunciation_result, transcript, topic)
            )
        finally:
            loop.close()
        
        # Argument analysis (stub)
        has_claim = True
        has_reason = True
        has_evidence = False
        argument_score = 75
        argument_summary = "Your argument has a clear claim and reasoning, but could benefit from more evidence."
        
        # Format pronunciation feedback
        pronunciation_feedback = []
        for correction in feedback.corrections:
            pronunciation_feedback.append(f"• **{correction.word}**: {correction.error_description} {correction.correction_tip}")
        
        pronunciation_text = "\n".join(pronunciation_feedback) if pronunciation_feedback else "No pronunciation errors detected!"
        
        return (
            transcript,
            has_claim,
            has_reason, 
            has_evidence,
            f"Score: {argument_score}/100",
            argument_summary,
            pronunciation_text
        )

# Initialize analyzer
analyzer = TurnAnalyzer()

def analyze_speech(audio, topic):
    if audio is None:
        return "Please record audio first.", False, False, False, "Score: 0/100", "No analysis available.", "Please record audio first."
    
    return analyzer.analyze(audio, topic)

# Create Gradio interface
with gr.Blocks(title="SpeakFlow AI - Pronunciation Coach") as demo:
    gr.Markdown("# SpeakFlow AI - Pronunciation Coach Prototype")
    gr.Markdown("Record your speech to get pronunciation feedback and argument analysis.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath",
                label="Record Your Speech"
            )
            topic_input = gr.Textbox(
                value="general discussion",
                label="Discussion Topic",
                placeholder="Enter the topic you're discussing..."
            )
            analyze_btn = gr.Button("Analyze Speech", variant="primary")
        
        with gr.Column(scale=2):
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=3,
                interactive=False
            )
            
            gr.Markdown("### Argument Analysis")
            with gr.Row():
                claim_check = gr.Checkbox(label="Has Claim", interactive=False)
                reason_check = gr.Checkbox(label="Has Reason", interactive=False)  
                evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
            
            argument_score = gr.Textbox(label="Argument Score", interactive=False)
            argument_summary = gr.Textbox(
                label="Argument Feedback", 
                lines=2,
                interactive=False
            )
            
            gr.Markdown("### Pronunciation Feedback")
            pronunciation_output = gr.Textbox(
                label="Mispronounced Words & Tips",
                lines=5,
                interactive=False
            )
    
    analyze_btn.click(
        fn=analyze_speech,
        inputs=[audio_input, topic_input],
        outputs=[
            transcript_output,
            claim_check,
            reason_check,
            evidence_check, 
            argument_score,
            argument_summary,
            pronunciation_output
        ]
    )

if __name__ == "__main__":
    demo.launch()