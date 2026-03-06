import gradio as gr
import asyncio
import tempfile
from pathlib import Path
from turn_analyzer import TurnAnalyzer, TurnInput

# Initialize the analyzer
analyzer = TurnAnalyzer()

async def analyze_turn(audio, transcript, topic, position, session_id, turn_number):
    """Process audio and transcript through TurnAnalyzer"""
    if audio is None or transcript.strip() == "":
        return (
            "No audio or transcript provided",
            False, False, False, 0.0, "No analysis performed",
            "No pronunciation errors detected"
        )
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio)
            audio_path = Path(temp_audio.name)
        
        # Create TurnInput
        turn_input = TurnInput(
            transcript=transcript,
            session_id=session_id or "demo_session",
            turn_number=int(turn_number) if turn_number else 1,
            topic=topic or "General Discussion",
            user_position=position,
            audio_path=audio_path
        )
        
        # Analyze the turn
        analysis = await analyzer.analyze(turn_input)
        
        # Format argument feedback
        arg_result = analysis.argument
        summary_text = f"Argument Score: {arg_result.argument_score:.2f}\n\nSummary: {arg_result.summary}"
        
        if arg_result.logical_gaps:
            summary_text += f"\n\nLogical Gaps:\n" + "\n".join([f"• {gap}" for gap in arg_result.logical_gaps])
        
        if arg_result.vocabulary_flags:
            summary_text += f"\n\nVocabulary Notes:\n" + "\n".join([f"• {flag}" for flag in arg_result.vocabulary_flags])
        
        # Format pronunciation feedback
        pron_result = analysis.pronunciation
        pron_text = f"Fluency Score: {pron_result.fluency_score:.2f}\n\n"
        
        if pron_result.mispronounced_words:
            pron_text += "Mispronounced Words:\n"
            for word_error in pron_result.mispronounced_words:
                pron_text += f"• {word_error.word} - Expected: {word_error.expected_ipa}, Actual: {word_error.actual_ipa} ({word_error.severity})\n"
        else:
            pron_text += "No pronunciation errors detected"
        
        if pron_result.target_phonemes:
            pron_text += f"\n\nTarget Phonemes to Practice: {', '.join(pron_result.target_phonemes)}"
        
        # Clean up temp file
        audio_path.unlink(missing_ok=True)
        
        return (
            transcript,
            arg_result.has_claim,
            arg_result.has_reasoning,
            arg_result.has_evidence,
            arg_result.argument_score,
            summary_text,
            pron_text
        )
        
    except Exception as e:
        return (
            f"Error: {str(e)}",
            False, False, False, 0.0, f"Analysis failed: {str(e)}",
            "Pronunciation analysis failed"
        )

def sync_analyze_turn(*args):
    """Synchronous wrapper for the async analyze function"""
    return asyncio.run(analyze_turn(*args))

# Create Gradio interface
with gr.Blocks(title="SpeakFlow AI - Turn Analyzer Demo") as demo:
    gr.Markdown("# SpeakFlow AI - Turn Analyzer Demo")
    gr.Markdown("Record audio and analyze debate turn structure and pronunciation")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="bytes",
                label="Record Audio"
            )
            transcript_input = gr.Textbox(
                label="Transcript",
                placeholder="Enter or edit transcript here...",
                lines=3
            )
            
            with gr.Row():
                topic_input = gr.Textbox(
                    label="Topic",
                    value="Climate Change Policy",
                    placeholder="Debate topic"
                )
                position_input = gr.Dropdown(
                    choices=["pro", "con"],
                    value="pro",
                    label="Position"
                )
            
            with gr.Row():
                session_input = gr.Textbox(
                    label="Session ID",
                    value="demo_001",
                    placeholder="Session identifier"
                )
                turn_input = gr.Number(
                    label="Turn Number",
                    value=1,
                    precision=0
                )
            
            analyze_btn = gr.Button("Analyze Turn", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("## Analysis Results")
            
            transcript_output = gr.Textbox(
                label="Processed Transcript",
                interactive=False
            )
            
            gr.Markdown("### Argument Feedback")
            with gr.Row():
                claim_check = gr.Checkbox(label="Has Claim", interactive=False)
                reasoning_check = gr.Checkbox(label="Has Reasoning", interactive=False)
                evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
            
            score_display = gr.Number(
                label="Argument Score",
                interactive=False,
                precision=2
            )
            
            summary_output = gr.Textbox(
                label="Argument Summary",
                lines=6,
                interactive=False
            )
            
            gr.Markdown("### Pronunciation Feedback")
            pronunciation_output = gr.Textbox(
                label="Pronunciation Analysis",
                lines=6,
                interactive=False
            )
    
    # Connect the analyze button to the function
    analyze_btn.click(
        fn=sync_analyze_turn,
        inputs=[
            audio_input,
            transcript_input,
            topic_input,
            position_input,
            session_input,
            turn_input
        ],
        outputs=[
            transcript_output,
            claim_check,
            reasoning_check,
            evidence_check,
            score_display,
            summary_output,
            pronunciation_output
        ]
    )

if __name__ == "__main__":
    demo.launch()