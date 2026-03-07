import gradio as gr
import os
os.environ["USE_STUB_MFA"] = "True"

from coach_policy import CoachPolicyAgent

def analyze_speech(audio_file):
    if audio_file is None:
        return "", False, False, False, 0, "", []
    
    agent = CoachPolicyAgent()
    result = agent.analyze(audio_file)
    
    transcript = result.get('transcript', '')
    
    argument_feedback = result.get('argument_feedback', {})
    has_claim = argument_feedback.get('has_claim', False)
    has_reasoning = argument_feedback.get('has_reasoning', False) 
    has_evidence = argument_feedback.get('has_evidence', False)
    argument_score = argument_feedback.get('score', 0)
    argument_summary = argument_feedback.get('summary', '')
    
    pronunciation_feedback = result.get('pronunciation_feedback', {})
    mispronounced_words = pronunciation_feedback.get('errors', [])
    
    pronunciation_display = []
    for error in mispronounced_words:
        word = error.get('word', '')
        expected_ipa = error.get('expected_ipa', '')
        actual_ipa = error.get('actual_ipa', '')
        pronunciation_display.append(f"{word}: expected [{expected_ipa}], got [{actual_ipa}]")
    
    return transcript, has_claim, has_reasoning, has_evidence, argument_score, argument_summary, pronunciation_display

with gr.Blocks(title="SpeakFlow AI Coach") as app:
    gr.Markdown("# SpeakFlow AI Coach Prototype")
    gr.Markdown("Record your speech to get argument structure and pronunciation feedback.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your speech"
            )
            analyze_btn = gr.Button("Analyze Speech", variant="primary")
        
        with gr.Column():
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=5,
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Argument Structure")
            claim_check = gr.Checkbox(label="Has Clear Claim", interactive=False)
            reasoning_check = gr.Checkbox(label="Has Reasoning", interactive=False)
            evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
            argument_score = gr.Number(label="Argument Score (0-10)", interactive=False)
            argument_summary = gr.Textbox(
                label="Argument Feedback",
                lines=3,
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("### Pronunciation Feedback")
            pronunciation_errors = gr.JSON(
                label="Mispronounced Words (with IPA)",
                show_label=True
            )
    
    analyze_btn.click(
        fn=analyze_speech,
        inputs=[audio_input],
        outputs=[
            transcript_output,
            claim_check,
            reasoning_check,
            evidence_check,
            argument_score,
            argument_summary,
            pronunciation_errors
        ]
    )

if __name__ == "__main__":
    app.launch()