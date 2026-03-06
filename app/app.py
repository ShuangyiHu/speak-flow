import gradio as gr
import os
from turn_analyzer import TurnAnalyzer

# Set stub mode for demo
os.environ['USE_STUB_MFA'] = 'True'

def analyze_turn(audio_file):
    if audio_file is None:
        return "", False, False, False, 0, "", []
    
    analyzer = TurnAnalyzer()
    result = analyzer.analyze(audio_file)
    
    transcript = result.get('transcript', '')
    
    # Extract argument components
    argument = result.get('argument', {})
    has_claim = argument.get('has_claim', False)
    has_reason = argument.get('has_reason', False) 
    has_evidence = argument.get('has_evidence', False)
    score = argument.get('score', 0)
    summary = argument.get('summary', '')
    
    # Extract pronunciation errors
    pronunciation = result.get('pronunciation', {})
    errors = pronunciation.get('errors', [])
    
    # Format pronunciation errors for display
    error_display = []
    for error in errors:
        word = error.get('word', '')
        ipa_actual = error.get('ipa_actual', '')
        ipa_expected = error.get('ipa_expected', '')
        error_display.append(f"{word}: {ipa_actual} → {ipa_expected}")
    
    return transcript, has_claim, has_reason, has_evidence, score, summary, error_display

with gr.Blocks(title="SpeakFlow AI - Turn Analyzer") as app:
    gr.Markdown("# SpeakFlow AI - Turn Analyzer")
    gr.Markdown("Record your speech to get transcript, argument analysis, and pronunciation feedback.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Audio"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=3,
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Argument Analysis")
            claim_check = gr.Checkbox(label="Has Claim", interactive=False)
            reason_check = gr.Checkbox(label="Has Reason", interactive=False)
            evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
            score_display = gr.Number(label="Argument Score", interactive=False)
            summary_box = gr.Textbox(
                label="Summary",
                lines=2,
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("### Pronunciation Feedback")
            pronunciation_list = gr.JSON(
                label="Mispronounced Words",
                container=True
            )
    
    analyze_btn.click(
        fn=analyze_turn,
        inputs=[audio_input],
        outputs=[
            transcript_box,
            claim_check,
            reason_check, 
            evidence_check,
            score_display,
            summary_box,
            pronunciation_list
        ]
    )

if __name__ == "__main__":
    app.launch()